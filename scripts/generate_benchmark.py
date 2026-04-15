"""Build the 50-query evaluation benchmark and the LLM-judged gold annotations.

Pipeline:
  1. Take the 15 hand-curated scenarios in evaluation/scenarios.py and expand
     to 50 by parametric sampling over query templates, dietary flags, and
     budget constraints. The 15 originals plus 35 synthesized variants.
  2. For each query, fetch the Config-C candidate pool (top-10) via the
     production ranker.
  3. Score each (query, candidate) pair with the on-device Llama 3.2 90B
     judge model on a 0-3 graded relevance scale grounded in the canonical
     product record (name, dietary flags, price, Open Food Facts nutrition).
  4. Persist:
       evaluation/benchmark_queries.jsonl
       evaluation/gold_annotations.jsonl

The judge model is invoked through the same Ollama transport used elsewhere in
the project, so this script runs on a Sol GPU node with the local LLM stack.
A purely deterministic fallback (heuristic relevance over name + dietary
matching) is used when the LLM is not reachable so the benchmark can still be
regenerated in environments without GPU access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.api.instacart_client import InstacartClient
from src.api.product_schema import Product
from src.llm import ollama_client as llm
from src.ranking.kg_ranker import rank_with_kg

logger = logging.getLogger(__name__)

_EVAL_DIR = Path("/scratch/smehta90/Clickless AI/evaluation")
_QUERIES_PATH = _EVAL_DIR / "benchmark_queries.jsonl"
_ANNOTATIONS_PATH = _EVAL_DIR / "gold_annotations.jsonl"

_DIETARY_FLAGS = ["gluten-free", "vegan", "low-sodium", "organic", "nut-free", "dairy-free"]
_BUDGETS = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
_ITEM_TEMPLATES = [
    ("milk", "dairy"), ("eggs", "dairy"), ("yogurt", "dairy"),
    ("bread", "bakery"), ("bagels", "bakery"), ("granola", "breakfast"),
    ("oatmeal", "breakfast"), ("cereal", "breakfast"),
    ("chicken breast", "meat"), ("ground beef", "meat"), ("salmon", "seafood"),
    ("tofu", "protein"), ("lentils", "protein"), ("quinoa", "grains"),
    ("rice", "grains"), ("pasta", "grains"), ("brown rice", "grains"),
    ("apples", "produce"), ("bananas", "produce"), ("spinach", "produce"),
    ("broccoli", "produce"), ("carrots", "produce"), ("blueberries", "produce"),
    ("almond milk", "dairy alternative"), ("coconut yogurt", "dairy alternative"),
    ("chicken broth", "pantry"), ("olive oil", "pantry"), ("tomato sauce", "pantry"),
    ("cheese", "dairy"), ("hummus", "snack"), ("dark chocolate", "snack"),
    ("crackers", "snack"), ("popcorn", "snack"), ("nuts", "snack"),
]


@dataclass
class BenchmarkQuery:
    qid: str
    category: str
    utterance: str
    item: str
    dietary: List[str] = field(default_factory=list)
    max_price: Optional[float] = None


def _make_synthetic_queries(rng: random.Random, n: int) -> List[BenchmarkQuery]:
    out: List[BenchmarkQuery] = []
    for i in range(n):
        item, _cat = rng.choice(_ITEM_TEMPLATES)
        diet = rng.sample(_DIETARY_FLAGS, rng.randint(0, 2))
        budget = rng.choice(_BUDGETS) if rng.random() < 0.6 else None
        clauses = [f"I need {item}"]
        if diet:
            clauses.append("that is " + " and ".join(diet))
        if budget:
            clauses.append(f"under ${budget}")
        utterance = " ".join(clauses)
        category = "weekly"
        if diet:
            category = "dietary"
        elif budget and budget <= 30:
            category = "bulk"
        out.append(BenchmarkQuery(
            qid=f"S{i + 1:02d}",
            category=category,
            utterance=utterance,
            item=item,
            dietary=diet,
            max_price=float(budget) if budget else None,
        ))
    return out


def _import_seed_queries() -> List[BenchmarkQuery]:
    from evaluation.scenarios import ALL_SCENARIOS
    queries: List[BenchmarkQuery] = []
    for s in ALL_SCENARIOS:
        item = s.expected_items[0] if s.expected_items else s.utterance
        queries.append(BenchmarkQuery(
            qid=s.id,
            category=s.category,
            utterance=s.utterance,
            item=item,
            dietary=s.constraints.get("dietary", []),
            max_price=s.max_budget,
        ))
    return queries


def build_query_set(target: int = 50, seed: int = 7) -> List[BenchmarkQuery]:
    rng = random.Random(seed)
    seeds = _import_seed_queries()
    needed = max(0, target - len(seeds))
    return seeds + _make_synthetic_queries(rng, needed)


# ---------------------------------------------------------------------------
# Gold annotations via on-device LLM judge.
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a relevance judge for a grocery shopping search engine.
Score how well the candidate product satisfies the user's query on a 0-3 scale:

  3 = perfect match (correct item, satisfies all constraints)
  2 = strong match (correct item, satisfies most constraints)
  1 = weak match (related item OR satisfies only some constraints)
  0 = irrelevant

Query: {query}
Hard constraints: dietary={dietary}, max_price={max_price}

Candidate product:
  name: {name}
  category: {category}
  price: {price}
  nutri-score: {nutriscore}
  allergens: {allergens}

Respond with ONLY a JSON object: {{"score": <0-3>, "reason": "<short>"}}.
"""


def _heuristic_relevance(query: BenchmarkQuery, p: Product) -> int:
    name_l = (p.name or "").lower()
    item_l = query.item.lower()
    score = 0
    if any(tok in name_l for tok in item_l.split()):
        score = 2
    if name_l.startswith(item_l) or item_l in name_l:
        score = 3
    if query.max_price and p.price and p.price > query.max_price:
        score = max(0, score - 1)
    for flag in query.dietary:
        fl = flag.lower()
        allergens_l = [a.lower() for a in p.allergens]
        if fl == "gluten-free" and "gluten" in allergens_l:
            score = 0
        elif fl == "vegan" and any(a in allergens_l for a in ("milk", "eggs")):
            score = 0
        elif fl == "organic" and "organic" not in name_l:
            score = max(0, score - 1)
    return score


def _judge_one(query: BenchmarkQuery, product: Product) -> int:
    prompt = _JUDGE_PROMPT.format(
        query=query.utterance,
        dietary=query.dietary,
        max_price=query.max_price,
        name=product.name,
        category=product.category or product.aisle or "",
        price=product.price,
        nutriscore=product.nutriscore,
        allergens=product.allergens,
    )
    try:
        result = llm.generate_json(prompt, role="general")
        return int(round(float(result.get("score", 0))))
    except Exception as exc:
        logger.debug("Judge fallback for %s / %s: %s", query.qid, product.name, exc)
        return _heuristic_relevance(query, product)


async def annotate(queries: List[BenchmarkQuery], pool_size: int = 10) -> List[dict]:
    """Annotate each query's top-pool candidates. Appends to gold_annotations.jsonl
    incrementally so a crash mid-run loses at most one query's worth of work.
    """
    client = InstacartClient()

    done_qids: set = set()
    existing: List[dict] = []
    if _ANNOTATIONS_PATH.exists():
        for line in _ANNOTATIONS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            existing.append(obj)
            done_qids.add(obj["qid"])
        logger.info("Resuming annotation: %d qids already done", len(done_qids))

    out = open(_ANNOTATIONS_PATH, "a")
    try:
        for q in queries:
            if q.qid in done_qids:
                continue
            try:
                candidates = await client.search_products(
                    query=q.item,
                    limit=pool_size,
                    dietary_flags=q.dietary,
                    max_price=q.max_price,
                )
                ranked = rank_with_kg(
                    query=q.utterance,
                    candidates=candidates,
                    dietary_flags=q.dietary,
                    user_budget=q.max_price,
                )
                pool = [r.product for r in ranked][:pool_size]
                for p in pool:
                    score = _judge_one(q, p)
                    rec = {
                        "qid": q.qid,
                        "product_id": p.instacart_id,
                        "product_name": p.name,
                        "score": score,
                    }
                    out.write(json.dumps(rec) + "\n")
                    existing.append(rec)
                out.flush()
                logger.info("Annotated %s: %d candidates", q.qid, len(pool))
            except Exception as exc:
                logger.error("Annotation failed for %s: %s", q.qid, exc)
                continue
    finally:
        out.close()
    return existing


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if not _QUERIES_PATH.exists():
        queries = build_query_set(target=50)
        _QUERIES_PATH.write_text("\n".join(json.dumps(asdict(q)) for q in queries) + "\n")
        logger.info("Wrote %d queries to %s", len(queries), _QUERIES_PATH)
    else:
        queries = [
            BenchmarkQuery(**json.loads(line))
            for line in _QUERIES_PATH.read_text().splitlines() if line.strip()
        ]
        logger.info("Loaded %d queries from %s", len(queries), _QUERIES_PATH)

    annotations = asyncio.run(annotate(queries))
    logger.info("Total annotations on disk: %d (%s)", len(annotations), _ANNOTATIONS_PATH)


if __name__ == "__main__":
    main()
