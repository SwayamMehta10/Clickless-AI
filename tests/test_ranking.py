"""Tests for classical baselines and KG ranker."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.product_schema import Product


def _data_available() -> bool:
    return (Path(__file__).resolve().parents[1] / "data" / "processed" / "transactions.pkl").exists()


def _model_available() -> bool:
    return (Path(__file__).resolve().parents[1] / "data" / "processed" / "logistic_ranker.pkl").exists()


@pytest.mark.skipif(not _data_available(), reason="transactions.pkl not found -- run preprocess_instacart.py")
def test_apriori_suggestions_banana():
    from src.ranking.apriori_miner import get_copurchase_suggestions
    suggestions = get_copurchase_suggestions("Banana", top_k=5)
    assert isinstance(suggestions, list)
    assert len(suggestions) <= 5


@pytest.mark.skipif(not _data_available(), reason="transactions.pkl not found")
def test_apriori_no_crash_unknown_product():
    from src.ranking.apriori_miner import get_copurchase_suggestions
    suggestions = get_copurchase_suggestions("xyznonexistent12345", top_k=5)
    assert isinstance(suggestions, list)


def test_logistic_ranker_ordering():
    """Logistic ranker should return products sorted by score descending."""
    from src.ranking.logistic_ranker import rank_products

    products = [
        Product(instacart_id="1", name="Organic Whole Milk", price=3.99, availability=True, reorder_rate=0.8),
        Product(instacart_id="2", name="Almond Milk", price=5.49, availability=True, reorder_rate=0.4),
        Product(instacart_id="3", name="Skim Milk", price=2.99, availability=False, reorder_rate=0.3),
    ]

    ranked = rank_products("organic milk", products)
    assert len(ranked) == 3
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_kg_ranker_hard_constraints():
    """KG ranker must exclude allergen-violating products when gluten-free is required."""
    from src.ranking.kg_ranker import rank_with_kg

    products = [
        Product(instacart_id="1", name="Whole Wheat Bread", price=3.0, allergens=["gluten", "wheat"]),
        Product(instacart_id="2", name="Gluten Free Bread", price=4.0, allergens=[]),
        Product(instacart_id="3", name="Sourdough", price=3.5, allergens=["gluten"]),
    ]

    ranked = rank_with_kg("bread", products, dietary_flags=["gluten-free"])
    product_ids = [r.product.instacart_id for r in ranked]
    assert "2" in product_ids
    assert "1" not in product_ids
    assert "3" not in product_ids


def test_kg_ranker_returns_ranked_products():
    from src.ranking.kg_ranker import rank_with_kg

    products = [
        Product(instacart_id=str(i), name=f"Product {i}", price=float(i), availability=True)
        for i in range(1, 6)
    ]
    ranked = rank_with_kg("grocery item", products)
    assert len(ranked) == 5
    assert all(r.rank == i + 1 for i, r in enumerate(ranked))
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
