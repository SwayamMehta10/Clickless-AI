"""Tests for NLU components (intent classifier, slot filler, dialogue state).

These tests run against the real Ollama models when available,
or skip gracefully when Ollama is not running.
"""

from __future__ import annotations

import pytest

from src.nlu.dialogue_state import DialogueState, Slots
from src.nlu.slot_filler import extract_slots, merge_slots


# ---------------------------------------------------------------------------
# DialogueState unit tests (no LLM required)
# ---------------------------------------------------------------------------

def test_dialogue_state_add_turn():
    ds = DialogueState()
    ds.add_turn("user", "hello")
    ds.add_turn("assistant", "hi there")
    assert len(ds.history) == 2
    assert ds.history[0].role == "user"
    assert ds.history[1].content == "hi there"


def test_dialogue_state_history_truncation():
    ds = DialogueState(max_history=3)
    for i in range(10):
        ds.add_turn("user", f"msg {i}")
    # max_history * 2 = 6 records kept
    assert len(ds.history) <= 6


def test_dialogue_state_cart_operations():
    from src.api.product_schema import CartItem, Product
    ds = DialogueState()
    product = Product(instacart_id="p1", name="Organic Milk", price=3.99)
    item = CartItem(product=product, quantity=2)

    ds.add_to_cart(item)
    assert len(ds.cart) == 1

    # Adding same product increases quantity
    ds.add_to_cart(CartItem(product=product, quantity=1))
    assert ds.cart[0].quantity == 3

    # Cart total
    assert ds.cart_total == pytest.approx(3.99 * 3, rel=1e-3)

    # Remove
    removed = ds.remove_from_cart("p1")
    assert removed
    assert len(ds.cart) == 0


def test_merge_slots():
    s1 = Slots(item="milk", quantity=2, unit="gallon")
    s2 = Slots(max_price=6.0, dietary_flags=["organic"])
    merged = merge_slots(s1, s2)
    assert merged.item == "milk"
    assert merged.quantity == 2
    assert merged.max_price == 6.0
    assert "organic" in merged.dietary_flags


def test_merge_slots_override():
    s1 = Slots(item="milk", quantity=2)
    s2 = Slots(item="eggs", quantity=12)
    merged = merge_slots(s1, s2)
    assert merged.item == "eggs"
    assert merged.quantity == 12


# ---------------------------------------------------------------------------
# Integration tests (require Ollama)
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        from src.llm.ollama_client import list_models
        models = list_models()
        return len(models) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_intent_classifier_search():
    from src.nlu.intent_classifier import classify
    intent, confidence = classify("I need 2 gallons of organic whole milk under $6")
    assert intent == "search_product"
    assert confidence > 0.7


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_intent_classifier_checkout():
    from src.nlu.intent_classifier import classify
    intent, _ = classify("Let's checkout")
    assert intent == "checkout"


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_slot_filler_canonical():
    """Verify the example from the implementation plan."""
    slots = extract_slots("I need 2 gallons of organic whole milk under $6")
    assert slots.item is not None and "milk" in slots.item.lower()
    assert slots.quantity == pytest.approx(2)
    assert slots.unit is not None and "gallon" in slots.unit.lower()
    assert slots.max_price == pytest.approx(6.0)
    assert "organic" in [f.lower() for f in slots.dietary_flags]
