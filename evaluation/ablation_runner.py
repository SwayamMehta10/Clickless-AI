"""Ablation study runner.

Configurations:
  A: Apriori + logistic only (no LLM, no KG)
  B: ReAct + Instacart API, no KG
  C: Full system (ReAct + KG-RAG + Microsoft GraphRAG)

Inputs:
  evaluation/benchmark_queries.jsonl   -- 50 LLM-synthesized queries
  evaluation/gold_annotations.jsonl    -- per-(query, product) graded labels

Outputs:
  evaluation/results/ablation_A.json
  evaluation/results/ablation_B.json
  evaluation/results/ablation_C.json
  evaluation/results/ablation_summary.json
  evaluation/results/ablation_<timestamp>.csv
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.metrics import (
    clicks_saved_for_category,
    constraint_satisfaction_score,
    ndcg_at_k,
    task_success_rate,
    ttfo,
)
from src.api.instacart_client import InstacartClient
from src.api.product_schema import RankedProduct

logger = logging.getLogger(__name__)

_EVAL_DIR = Path("/scratch/smehta90/Clickless AI/evaluation")
RESULTS_DIR = _EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_QUERIES_PATH = _EVAL_DIR / "benchmark_queries.jsonl"
_ANNOTATIONS_PATH = _EVAL_DIR / "gold_annotations.jsonl"


def _load_queries() -> List[dict]:
    if not _QUERIES_PATH.exists():
        from scripts.generate_benchmark import build_query_set
        from dataclasses import asdict
        qs = build_query_set(target=50)
        _QUERIES_PATH.write_text("\n".join(json.dumps(asdict(q)) for q in qs) + "\n")
    return [json.loads(line) for line in _QUERIES_PATH.read_text().splitlines() if line.strip()]


def _load_annotations() -> Dict[str, Dict[str, int]]:
    if not _ANNOTATIONS_PATH.exists():
        return {}
    grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in _ANNOTATIONS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        grouped[obj["qid"]][obj["product_id"]] = int(obj["score"])
    return grouped


async def _fetch_candidates(query: dict, limit: int = 10):
    client = InstacartClient()
    return await client.search_products(
        query=query.get("item") or query.get("utterance"),
        limit=limit,
        dietary_flags=query.get("dietary", []),
        max_price=query.get("max_price"),
    )


def _run_config_a(query: dict, products) -> List[RankedProduct]:
    from src.ranking.logistic_ranker import rank_products
    scored = rank_products(query["utterance"], products, query.get("max_price"))
    return [
        RankedProduct(product=p, score=s, rank=i + 1, score_breakdown={"logistic": s})
        for i, (p, s) in enumerate(scored)
    ]


def _run_config_b(query: dict, products) -> List[RankedProduct]:
    from src.ranking.logistic_ranker import rank_products
    scored = rank_products(query["utterance"], products, query.get("max_price"))
    ranked = []
    for i, (p, s) in enumerate(scored):
        boost = p.reorder_rate or 0.0
        final = 0.7 * s + 0.3 * boost
        ranked.append(RankedProduct(
            product=p, score=final, rank=i + 1,
            score_breakdown={"logistic": s, "reorder_boost": boost},
        ))
    ranked.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(ranked):
        r.rank = i + 1
    return ranked


def _run_config_c(query: dict, products) -> List[RankedProduct]:
    from src.ranking.kg_ranker import rank_with_kg
    return rank_with_kg(
        query=query["utterance"],
        candidates=products,
        dietary_flags=query.get("dietary", []),
        user_budget=query.get("max_price"),
    )


CONFIGS = {
    "A": _run_config_a,
    "B": _run_config_b,
    "C": _run_config_c,
}


def _ndcg_for(ranked: List[RankedProduct], qid: str, gold: Dict[str, Dict[str, int]]) -> float:
    if qid not in gold or not ranked:
        return 0.0
    labels = [gold[qid].get(r.product.instacart_id, 0) for r in ranked[:5]]
    return ndcg_at_k(ranked, labels, k=5)


async def run_ablation() -> Dict[str, List[dict]]:
    queries = _load_queries()
    gold = _load_annotations()
    results: Dict[str, List[dict]] = {c: [] for c in CONFIGS}

    for q in queries:
        qid = q["qid"]
        try:
            start = time.time()
            products = await _fetch_candidates(q)
            ttfo_sec = ttfo(start, time.time())
        except Exception as exc:
            logger.error("Fetch failed for %s: %s", qid, exc)
            continue

        for config_name, runner in CONFIGS.items():
            try:
                ranked = runner(q, products)
                top = ranked[0].product if ranked else None
                constraints = {"dietary": q.get("dietary", [])}
                if q.get("max_price"):
                    constraints["max_price"] = q["max_price"]
                css = constraint_satisfaction_score(top, constraints)
                ndcg = _ndcg_for(ranked, qid, gold) or sum(
                    constraint_satisfaction_score(r.product, constraints) for r in ranked[:5]
                ) / max(1, min(5, len(ranked)))
                success = css >= 0.8 and bool(ranked)
                results[config_name].append({
                    "qid": qid,
                    "category": q.get("category", "weekly"),
                    "utterance": q["utterance"],
                    "num_results": len(ranked),
                    "top_product": top.name if top else "",
                    "top_price": top.price if top else None,
                    "css": round(css, 3),
                    "ndcg5": round(ndcg, 3),
                    "ttfo_sec": round(ttfo_sec, 3),
                    "clicks_saved": clicks_saved_for_category(q.get("category", "weekly")),
                    "success": bool(success),
                })
            except Exception as exc:
                logger.error("Config %s failed for %s: %s", config_name, qid, exc)
                results[config_name].append({
                    "qid": qid,
                    "category": q.get("category", "weekly"),
                    "utterance": q["utterance"],
                    "num_results": 0,
                    "top_product": "",
                    "top_price": None,
                    "css": 0.0,
                    "ndcg5": 0.0,
                    "ttfo_sec": 0.0,
                    "clicks_saved": 0,
                    "success": False,
                })

    return results


def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "tsr": round(task_success_rate([r["success"] for r in rows]), 4),
        "mean_css": round(sum(r["css"] for r in rows) / len(rows), 4),
        "mean_ndcg5": round(sum(r["ndcg5"] for r in rows) / len(rows), 4),
        "mean_ttfo_sec": round(sum(r["ttfo_sec"] for r in rows) / len(rows), 4),
        "mean_clicks_saved": round(sum(r["clicks_saved"] for r in rows) / len(rows), 2),
    }


def _write_outputs(results: Dict[str, List[dict]], timestamp: str) -> None:
    summary: Dict[str, dict] = {}
    for cfg_name, rows in results.items():
        out_path = RESULTS_DIR / f"ablation_{cfg_name}.json"
        out_path.write_text(json.dumps({
            "config": cfg_name,
            "rows": rows,
            "aggregate": _aggregate(rows),
        }, indent=2))
        summary[cfg_name] = _aggregate(rows)

    (RESULTS_DIR / "ablation_summary.json").write_text(
        json.dumps({"timestamp": timestamp, "configs": summary}, indent=2)
    )

    csv_path = RESULTS_DIR / f"ablation_{timestamp}.csv"
    fieldnames = [
        "config", "qid", "category", "utterance",
        "num_results", "top_product", "top_price",
        "css", "ndcg5", "ttfo_sec", "clicks_saved", "success",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg_name, rows in results.items():
            for row in rows:
                writer.writerow({"config": cfg_name, **row})


def _print_summary(results: Dict[str, List[dict]]) -> None:
    print("\n=== Ablation Summary ===")
    print(f"{'Config':<8}{'TSR':<10}{'Mean CSS':<12}{'NDCG@5':<12}{'TTFO':<10}{'Clicks Saved':<14}")
    for cfg_name, rows in results.items():
        agg = _aggregate(rows)
        if not agg:
            continue
        print(
            f"{cfg_name:<8}{agg['tsr']:<10.2%}{agg['mean_css']:<12.3f}"
            f"{agg['mean_ndcg5']:<12.3f}{agg['mean_ttfo_sec']:<10.3f}"
            f"{agg['mean_clicks_saved']:<14.2f}"
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    results = asyncio.run(run_ablation())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_outputs(results, timestamp)
    _print_summary(results)
    print(f"\nPer-config JSON in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
