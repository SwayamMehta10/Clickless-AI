"""Ablation study runner.

Configurations:
  A: Apriori + logistic only (no LLM, no KG)
  B: ReAct + Instacart API, no KG
  C: Full system (ReAct + KG-RAG + GraphRAG)

Produces evaluation/results/ablation_<timestamp>.csv
"""

from __future__ import annotations

import asyncio
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    constraint_satisfaction_score,
    ndcg_at_k,
    task_success_rate,
    ttfo,
)
from evaluation.scenarios import ALL_SCENARIOS, Scenario
from src.api.instacart_mock import MockInstacartClient
from src.api.product_schema import RankedProduct

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def _fetch_candidates(query: str, scenario: Scenario, limit: int = 10):
    client = MockInstacartClient()
    max_price = scenario.max_budget
    dietary = scenario.constraints.get("dietary", [])
    return await client.search_products(
        query=query,
        limit=limit,
        dietary_flags=dietary,
        max_price=max_price,
    )


def _run_config_a(scenario: Scenario, products) -> List[RankedProduct]:
    """Apriori + logistic only."""
    from src.ranking.logistic_ranker import rank_products
    scored = rank_products(scenario.utterance, products, scenario.max_budget)
    ranked = [
        RankedProduct(product=p, score=s, rank=i + 1, score_breakdown={"logistic": s})
        for i, (p, s) in enumerate(scored)
    ]
    return ranked


def _run_config_b(scenario: Scenario, products) -> List[RankedProduct]:
    """ReAct + API, no KG (uses logistic + simple reorder)."""
    from src.ranking.logistic_ranker import rank_products
    scored = rank_products(scenario.utterance, products, scenario.max_budget)
    ranked = []
    for i, (p, s) in enumerate(scored):
        # Boost by reorder rate if available
        boost = p.reorder_rate or 0.0
        final_score = 0.7 * s + 0.3 * boost
        ranked.append(RankedProduct(
            product=p, score=final_score, rank=i + 1,
            score_breakdown={"logistic": s, "reorder_boost": boost},
        ))
    ranked.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(ranked):
        r.rank = i + 1
    return ranked


def _run_config_c(scenario: Scenario, products) -> List[RankedProduct]:
    """Full system with KG-RAG."""
    from src.ranking.kg_ranker import rank_with_kg
    return rank_with_kg(
        query=scenario.utterance,
        candidates=products,
        dietary_flags=scenario.constraints.get("dietary", []),
        user_budget=scenario.max_budget,
    )


CONFIGS = {
    "A": _run_config_a,
    "B": _run_config_b,
    "C": _run_config_c,
}


async def run_ablation() -> Dict[str, List[dict]]:
    results: Dict[str, List[dict]] = {c: [] for c in CONFIGS}

    for scenario in ALL_SCENARIOS:
        logger.info("Running scenario %s (%s)", scenario.id, scenario.category)

        try:
            start = time.time()
            products = await _fetch_candidates(scenario.utterance, scenario)
            ttfo_sec = ttfo(start, time.time())
        except Exception as exc:
            logger.error("Fetch failed for %s: %s", scenario.id, exc)
            continue

        for config_name, runner in CONFIGS.items():
            try:
                ranked = runner(scenario, products)
                top = ranked[0].product if ranked else None

                # Build constraints dict for CSS
                constraints = {"dietary": scenario.constraints.get("dietary", [])}
                if scenario.max_budget:
                    constraints["max_price"] = scenario.max_budget

                css = constraint_satisfaction_score(top, constraints)
                success = css >= 0.8 and ranked

                # NDCG: use a simple relevance label based on CSS of each result
                rel_labels = [
                    constraint_satisfaction_score(r.product, constraints)
                    for r in ranked[:10]
                ]
                ndcg = ndcg_at_k(ranked, rel_labels, k=5)

                results[config_name].append({
                    "scenario_id": scenario.id,
                    "category": scenario.category,
                    "utterance": scenario.utterance,
                    "num_results": len(ranked),
                    "top_product": top.name if top else "",
                    "top_price": top.price if top else None,
                    "css": round(css, 3),
                    "ndcg5": round(ndcg, 3),
                    "ttfo_sec": round(ttfo_sec, 3),
                    "success": bool(success),
                })
            except Exception as exc:
                logger.error("Config %s failed for %s: %s", config_name, scenario.id, exc)
                results[config_name].append({
                    "scenario_id": scenario.id,
                    "category": scenario.category,
                    "utterance": scenario.utterance,
                    "num_results": 0,
                    "top_product": "",
                    "top_price": None,
                    "css": 0.0,
                    "ndcg5": 0.0,
                    "ttfo_sec": 0.0,
                    "success": False,
                })

    return results


def _write_csv(results: Dict[str, List[dict]], timestamp: str) -> Path:
    out_path = RESULTS_DIR / f"ablation_{timestamp}.csv"
    fieldnames = [
        "config", "scenario_id", "category", "utterance",
        "num_results", "top_product", "top_price",
        "css", "ndcg5", "ttfo_sec", "success",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_name, rows in results.items():
            for row in rows:
                writer.writerow({"config": config_name, **row})
    return out_path


def _print_summary(results: Dict[str, List[dict]]) -> None:
    print("\n=== Ablation Summary ===")
    print(f"{'Config':<8}{'TSR':<10}{'Mean CSS':<12}{'Mean NDCG@5':<14}{'Mean TTFO':<12}")
    for config_name, rows in results.items():
        if not rows:
            continue
        tsr = task_success_rate([r["success"] for r in rows])
        mean_css = sum(r["css"] for r in rows) / len(rows)
        mean_ndcg = sum(r["ndcg5"] for r in rows) / len(rows)
        mean_ttfo = sum(r["ttfo_sec"] for r in rows) / len(rows)
        print(f"{config_name:<8}{tsr:<10.2%}{mean_css:<12.3f}{mean_ndcg:<14.3f}{mean_ttfo:<12.3f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    results = asyncio.run(run_ablation())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _write_csv(results, timestamp)
    _print_summary(results)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
