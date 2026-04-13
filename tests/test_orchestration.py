"""Tests for LangGraph orchestration pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.api.product_schema import CartItem, Product
from src.nlu.dialogue_state import DialogueState


def _make_product(id_: str, name: str, price: float = 3.99) -> Product:
    return Product(instacart_id=id_, name=name, price=price, availability=True)


def test_state_initial_values():
    from src.orchestration.state import AgentState
    ds = DialogueState()
    state: AgentState = {
        "messages": [],
        "dialogue_state": ds,
        "search_results": [],
        "ranked_results": [],
        "cart": [],
        "checkout_ready": False,
        "error": None,
        "user_id": "test",
        "session_id": "s1",
        "response_text": None,
    }
    assert state["checkout_ready"] is False
    assert state["error"] is None


@patch("src.orchestration.agents.intent_classifier.classify", return_value=("search_product", 0.95))
@patch("src.orchestration.agents.slot_filler.extract_slots")
@patch("src.orchestration.agents.get_client")
def test_nlu_agent_sets_intent(mock_client, mock_slots, mock_classify):
    from src.nlu.dialogue_state import Slots
    from src.orchestration.agents import nlu_agent
    from src.orchestration.state import AgentState

    mock_slots.return_value = Slots(item="milk", max_price=6.0)

    ds = DialogueState()
    state: AgentState = {
        "messages": [HumanMessage(content="I need organic milk under $6")],
        "dialogue_state": ds,
        "search_results": [],
        "ranked_results": [],
        "cart": [],
        "checkout_ready": False,
        "error": None,
        "user_id": "test",
        "session_id": "s1",
        "response_text": None,
    }

    result = nlu_agent(state)
    assert result["dialogue_state"].current_intent == "search_product"


@patch("src.orchestration.agents.intent_classifier.classify", return_value=("checkout", 0.99))
def test_routing_checkout(_):
    from src.orchestration.graph_builder import _route_after_nlu
    from src.orchestration.state import AgentState

    ds = DialogueState()
    ds.current_intent = "checkout"
    state: AgentState = {
        "messages": [],
        "dialogue_state": ds,
        "search_results": [],
        "ranked_results": [],
        "cart": [],
        "checkout_ready": False,
        "error": None,
        "user_id": "test",
        "session_id": "s1",
        "response_text": None,
    }
    assert _route_after_nlu(state) == "checkout_handoff"


def test_checkout_handoff_empty_cart():
    from src.orchestration.agents import checkout_handoff
    from src.orchestration.state import AgentState

    ds = DialogueState()
    state: AgentState = {
        "messages": [],
        "dialogue_state": ds,
        "search_results": [],
        "ranked_results": [],
        "cart": [],
        "checkout_ready": False,
        "error": None,
        "user_id": "test",
        "session_id": "s1",
        "response_text": None,
    }
    result = checkout_handoff(state)
    assert result["error"] is not None
    assert result["checkout_ready"] is False


def test_checkout_handoff_with_items():
    from src.orchestration.agents import checkout_handoff
    from src.orchestration.state import AgentState

    ds = DialogueState()
    product = _make_product("1", "Milk")
    ds.add_to_cart(CartItem(product=product, quantity=1))

    state: AgentState = {
        "messages": [],
        "dialogue_state": ds,
        "search_results": [],
        "ranked_results": [],
        "cart": ds.cart,
        "checkout_ready": False,
        "error": None,
        "user_id": "test",
        "session_id": "s1",
        "response_text": None,
    }
    result = checkout_handoff(state)
    assert result["checkout_ready"] is True
    assert result["error"] is None
