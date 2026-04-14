"""KG-enriched product ranker.

Combines: logistic score + KG nutritional match + Apriori co-purchase + GraphRAG relevance.
Weighted linear combination with configurable weights.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from src.api.product_schema import Product, RankedProduct
from src.knowledge_graph.graphrag_interface import get_relevance_score
from src.ranking.apriori_miner import get_copurchase_suggestions
from src.ranking.logistic_ranker import rank_products as logistic_rank
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {
    "logistic_score": 0.35,
    "kg_nutrition_match": 0.25,
    "apriori_copurchase": 0.20,
    "graphrag_relevance": 0.20,
}


def _nutrition_score(product: Product, dietary_flags: List[str]) -> float:
    """Score based on Nutri-Score and NOVA group."""
    ns_scores = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "unknown": 0.5}
    ns = str(product.nutriscore).upper() if product.nutriscore else "unknown"
    base = ns_scores.get(ns, 0.5)

    # Bonus/penalty based on dietary flags vs allergens
    allergens_l = [a.lower() for a in product.allergens]
    for flag in dietary_flags:
        fl = flag.lower()
        if fl == "gluten-free" and "gluten" in allergens_l:
            return 0.0  # Hard constraint violated
        if fl == "vegan" and any(a in allergens_l for a in ["milk", "eggs"]):
            return 0.0
        if fl == "low-sodium" and product.sodium_mg and product.sodium_mg > 600:
            base *= 0.5

    return base


def _apriori_score(product: Product, cart_items: List[str]) -> float:
    """Score 1.0 if product appears in co-purchase suggestions for any cart item, else 0.0."""
    if not cart_items:
        return 0.5  # Neutral when cart is empty
    for cart_item in cart_items:
        suggestions = get_copurchase_suggestions(cart_item, top_k=10)
        if any(product.name and product.name.lower() in s.lower() for s in suggestions):
            return 1.0
    return 0.0


def rank_with_kg(
    query: str,
    candidates: List[Product],
    dietary_flags: Optional[List[str]] = None,
    user_budget: Optional[float] = None,
    cart_item_names: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    relevance_scorer: Optional[Callable[[str, str, Optional[List[str]]], float]] = None,
) -> List[RankedProduct]:
    """Rank products using all available signals.

    Returns a list of RankedProduct sorted by composite score descending.
    """
    if not candidates:
        return []

    dietary_flags = dietary_flags or []
    cart_item_names = cart_item_names or []

    cfg = get_settings()
    w = weights or cfg["ranking"]["kg_ranker"]["weights"]

    # 1. Logistic scores
    logistic_scored = logistic_rank(query, candidates, user_budget)
    logistic_map = {p.instacart_id: score for p, score in logistic_scored}

    ranked = []
    for product in candidates:
        # Hard constraint: skip products that violate dietary flags
        if dietary_flags:
            allergens_l = [a.lower() for a in product.allergens]
            skip = False
            for flag in dietary_flags:
                fl = flag.lower()
                if fl == "gluten-free" and "gluten" in allergens_l:
                    skip = True
                if fl == "vegan" and any(a in allergens_l for a in ["milk", "eggs"]):
                    skip = True
            if skip:
                continue

        # Compute component scores
        logistic = logistic_map.get(product.instacart_id, 0.5)
        nutrition = _nutrition_score(product, dietary_flags)
        apriori = _apriori_score(product, cart_item_names)

        # GraphRAG relevance (skip if KG unavailable to keep latency low)
        try:
            scorer = relevance_scorer or get_relevance_score
            graphrag = scorer(product.name or "", query, dietary_flags)
        except Exception:
            graphrag = 0.3

        composite = (
            w["logistic_score"] * logistic
            + w["kg_nutrition_match"] * nutrition
            + w["apriori_copurchase"] * apriori
            + w["graphrag_relevance"] * graphrag
        )

        breakdown = {
            "logistic": round(logistic, 3),
            "nutrition_kg": round(nutrition, 3),
            "apriori": round(apriori, 3),
            "graphrag": round(graphrag, 3),
            "composite": round(composite, 3),
        }

        # Fetch co-purchase suggestions for display
        copurchase = []
        try:
            if product.name:
                copurchase = get_copurchase_suggestions(product.name, top_k=3)
        except Exception:
            pass

        ranked.append(RankedProduct(
            product=product,
            score=composite,
            rank=0,  # Set after sorting
            score_breakdown=breakdown,
            copurchase_suggestions=copurchase,
        ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked
