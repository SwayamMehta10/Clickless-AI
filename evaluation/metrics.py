"""Evaluation metrics: TSR, CSS, NDCG@5, Clicks Saved, TTFO."""

from __future__ import annotations

import math
from typing import List, Optional

from src.api.product_schema import Product, RankedProduct


def task_success_rate(results: List[bool]) -> float:
    """Fraction of scenarios where all hard constraints were met and a valid result was returned."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def constraint_satisfaction_score(
    top_product: Optional[Product],
    constraints: dict,
) -> float:
    """Fraction of constraints in the top result."""
    if top_product is None:
        return 0.0

    total = 0
    met = 0

    # Budget constraint
    if "max_price" in constraints:
        total += 1
        if top_product.price is None or top_product.price <= constraints["max_price"]:
            met += 1

    # Dietary constraints
    for flag in constraints.get("dietary", []):
        total += 1
        allergens = [a.lower() for a in top_product.allergens]
        if flag == "gluten-free" and "gluten" not in allergens:
            met += 1
        elif flag == "vegan" and not any(a in allergens for a in ["milk", "eggs"]):
            met += 1
        elif flag == "organic" and "organic" in (top_product.name or "").lower():
            met += 1
        elif flag == "nut-free" and not any("nut" in a for a in allergens):
            met += 1
        else:
            # Unknown flag -- give benefit of the doubt
            met += 1

    # Availability
    total += 1
    if top_product.availability:
        met += 1

    return met / total if total else 0.0


def ndcg_at_k(ranked: List[RankedProduct], relevance_labels: List[float], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not ranked or not relevance_labels:
        return 0.0

    k = min(k, len(ranked), len(relevance_labels))
    dcg = sum(
        relevance_labels[i] / math.log2(i + 2)
        for i in range(k)
    )
    ideal_labels = sorted(relevance_labels, reverse=True)[:k]
    idcg = sum(
        ideal_labels[i] / math.log2(i + 2)
        for i in range(k)
    )
    return dcg / idcg if idcg > 0 else 0.0


def clicks_saved(manual_clicks: int, agent_clicks: int) -> int:
    return max(0, manual_clicks - agent_clicks)


def ttfo(start_ts: float, first_result_ts: float) -> float:
    """Time to First Option in seconds."""
    return first_result_ts - start_ts
