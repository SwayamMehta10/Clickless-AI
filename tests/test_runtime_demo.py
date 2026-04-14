"""Tests for session-scoped demo runtime controls and mock adapters."""

from __future__ import annotations

import pytest

from src.api.product_schema import CartItem, Product
from src.nlu.dialogue_state import DialogueState
from src.runtime import demo as runtime_demo


def test_apply_service_mode_accepts_ready_live(monkeypatch):
    config = runtime_demo.DemoRuntimeConfig.all_mocked()
    monkeypatch.setitem(
        runtime_demo._LIVE_READINESS,
        runtime_demo.SHOPPING_API,
        lambda: runtime_demo.LiveReadiness(True, "ready"),
    )

    updated, error = runtime_demo.apply_service_mode(config, runtime_demo.SHOPPING_API, runtime_demo.LIVE)
    assert error is None
    assert updated.mode_for(runtime_demo.SHOPPING_API) == runtime_demo.LIVE


def test_apply_service_mode_rejects_unready_live(monkeypatch):
    config = runtime_demo.DemoRuntimeConfig.all_mocked()
    monkeypatch.setitem(
        runtime_demo._LIVE_READINESS,
        runtime_demo.SHOPPING_API,
        lambda: runtime_demo.LiveReadiness(False, "missing key"),
    )

    updated, error = runtime_demo.apply_service_mode(config, runtime_demo.SHOPPING_API, runtime_demo.LIVE)
    assert updated.mode_for(runtime_demo.SHOPPING_API) == runtime_demo.MOCKED
    assert error is not None


def test_apply_preset_all_live_partial(monkeypatch):
    readiness = {
        runtime_demo.SHOPPING_API: runtime_demo.LiveReadiness(True, "ready"),
        runtime_demo.LLM: runtime_demo.LiveReadiness(False, "ollama missing"),
        runtime_demo.KNOWLEDGE_GRAPH: runtime_demo.LiveReadiness(True, "ready"),
        runtime_demo.RANKING: runtime_demo.LiveReadiness(False, "artifacts missing"),
        runtime_demo.CHECKOUT: runtime_demo.LiveReadiness(True, "ready"),
    }
    monkeypatch.setattr(runtime_demo, "get_live_readiness", lambda service: readiness[service])

    config, errors = runtime_demo.apply_preset_selection(runtime_demo.DemoRuntimeConfig.all_mocked(), "All Live")
    assert config.mode_for(runtime_demo.SHOPPING_API) == runtime_demo.LIVE
    assert config.mode_for(runtime_demo.LLM) == runtime_demo.MOCKED
    assert config.mode_for(runtime_demo.KNOWLEDGE_GRAPH) == runtime_demo.LIVE
    assert config.mode_for(runtime_demo.RANKING) == runtime_demo.MOCKED
    assert config.mode_for(runtime_demo.CHECKOUT) == runtime_demo.LIVE
    assert len(errors) == 2


def test_mock_llm_classification_and_slots():
    intent, confidence = runtime_demo.classify_intent_mock("I need gluten-free bread under $5")
    slots = runtime_demo.extract_slots_mock("I need 2 gallons of organic whole milk under $6")
    assert intent == "search_product"
    assert confidence > 0.5
    assert slots.item is not None and "whole milk" in slots.item
    assert slots.max_price == pytest.approx(6.0)
    assert "organic" in slots.dietary_flags


def test_mock_ranker_returns_ranked_products():
    candidates = [
        Product(instacart_id="1", name="Organic Whole Milk", price=3.5, availability=True, reorder_rate=0.8),
        Product(instacart_id="2", name="Bananas", price=1.5, availability=True, reorder_rate=0.7),
        Product(instacart_id="3", name="Whole Wheat Bread", price=2.5, availability=True, allergens=["gluten"]),
    ]
    ranked = runtime_demo.mock_rank_products("organic milk", candidates, dietary_flags=[], user_budget=5.0)
    assert len(ranked) == 3
    assert ranked[0].product.name == "Organic Whole Milk"
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))


def test_mock_kg_returns_nonempty_subgraph():
    subgraph = runtime_demo.get_product_subgraph_mock("Organic Whole Milk")
    assert subgraph["nodes"]
    assert subgraph["edges"]


@pytest.mark.asyncio
async def test_mock_checkout_returns_success():
    result = await runtime_demo.run_checkout_mock(
        [CartItem(product=Product(instacart_id="1", name="Milk", price=3.99), quantity=2)]
    )
    assert result["success"] is True
    assert result["items"] == 1


def test_run_pipeline_all_mocked_smoke():
    from src.orchestration.graph_builder import run_pipeline

    dialogue_state = DialogueState(session_id="demo")
    runtime_config = runtime_demo.DemoRuntimeConfig.all_mocked()
    result = run_pipeline(
        user_message="I need milk",
        dialogue_state=dialogue_state,
        user_id="demo-user",
        session_id="demo-session",
        runtime_config=runtime_config,
    )

    assert result["response_text"]
    assert result["ranked_results"]
    assert result["runtime_config"] == runtime_config
