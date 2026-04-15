"""LangGraph agent node implementations.

Each agent is a function (AgentState) -> AgentState.
ReAct-style: Thought -> Action -> Observation loop within each node.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from src.api.instacart_client import get_client
from src.api.product_schema import CartItem
from src.llm import ollama_client as llm
from src.nlu import intent_classifier, slot_filler
from src.nlu.dialogue_state import DialogueState, Slots
from src.orchestration.state import AgentState
from src.ranking.kg_ranker import rank_with_kg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# NLU Agent
# ---------------------------------------------------------------------------

def nlu_agent(state: AgentState) -> AgentState:
    """Classify intent and extract slots from the latest user message."""
    messages = state["messages"]
    if not messages:
        return state

    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return state

    ds: DialogueState = state["dialogue_state"]
    history_text = ds.get_history_text()

    # Intent classification
    intent, confidence = intent_classifier.classify(last_user_msg, history_text)
    logger.info("NLU: intent=%s (confidence=%.2f)", intent, confidence)

    # Slot filling (only for search/add/set_constraint intents)
    new_slots = Slots()
    if intent in ("search_product", "add_to_cart", "set_constraint"):
        new_slots = slot_filler.extract_slots(last_user_msg, history_text)
        ds.slots = slot_filler.merge_slots(ds.slots, new_slots)

    ds.current_intent = intent
    ds.add_turn("user", last_user_msg)

    return {**state, "dialogue_state": ds}


# ---------------------------------------------------------------------------
# API Agent
# ---------------------------------------------------------------------------

def api_agent(state: AgentState) -> AgentState:
    """Search for products via Instacart API (or mock)."""
    ds: DialogueState = state["dialogue_state"]

    if ds.current_intent not in ("search_product", "get_recommendation"):
        return state

    query = ds.slots.item or ""
    if not query:
        return {**state, "error": "No search query extracted from your message."}

    client = get_client()
    try:
        products = _run_async(client.search_products(
            query=query,
            zip_code=ds.zip_code,
            limit=20,
            dietary_flags=ds.slots.dietary_flags or ds.dietary_preferences,
            max_price=ds.slots.max_price or ds.budget,
        ))
        logger.info("API: found %d products for query '%s'", len(products), query)
    except Exception as exc:
        logger.error("API search failed: %s", exc)
        return {**state, "error": f"Search failed: {exc}", "search_results": []}

    return {**state, "search_results": products}


# ---------------------------------------------------------------------------
# KG Ranking Agent
# ---------------------------------------------------------------------------

def kg_ranking_agent(state: AgentState) -> AgentState:
    """Rank search results using the full KG-enriched pipeline."""
    products = state.get("search_results", [])
    if not products:
        return state

    ds: DialogueState = state["dialogue_state"]
    cart_names = [item.product.name for item in ds.cart if item.product.name]

    ranked = rank_with_kg(
        query=ds.slots.item or "",
        candidates=products,
        dietary_flags=ds.slots.dietary_flags or ds.dietary_preferences,
        user_budget=ds.slots.max_price or ds.budget,
        cart_item_names=cart_names,
    )
    logger.info("Ranking: %d products ranked", len(ranked))
    return {**state, "ranked_results": ranked}


# ---------------------------------------------------------------------------
# Cart Agent
# ---------------------------------------------------------------------------

def cart_agent(state: AgentState) -> AgentState:
    """Handle add_to_cart and remove_from_cart intents."""
    ds: DialogueState = state["dialogue_state"]

    if ds.current_intent == "add_to_cart":
        ranked = state.get("ranked_results", [])
        search = state.get("search_results", [])

        # Default to top ranked product if available, else first search result
        candidates = [r.product for r in ranked] if ranked else search
        if candidates:
            product = candidates[0]
            qty = int(ds.slots.quantity or 1)
            item = CartItem(product=product, quantity=qty)
            ds.add_to_cart(item)
            logger.info("Cart: added %s x%d", product.name, qty)
        else:
            return {**state, "error": "No product found to add to cart."}

    elif ds.current_intent == "remove_from_cart":
        # Try to match by item name from slots
        item_name = (ds.slots.item or "").lower()
        removed = False
        for cart_item in list(ds.cart):
            if item_name and item_name in (cart_item.product.name or "").lower():
                ds.remove_from_cart(cart_item.product.instacart_id)
                removed = True
                break
        if not removed:
            return {**state, "error": f"Item '{item_name}' not found in cart."}

    # Sync cart to state
    return {**state, "cart": ds.cart, "dialogue_state": ds}


# ---------------------------------------------------------------------------
# Response Generator
# ---------------------------------------------------------------------------

_RESPONSE_PROMPT = """\
You are a helpful grocery shopping assistant for ClickLess AI.
Generate a concise, friendly response based on the current state.

Intent: {intent}
Slots: {slots}
Search results (top 3): {results}
Cart ({cart_count} items, total: ${cart_total}): {cart_summary}
Error: {error}

Keep response under 3 sentences. Mention specific product names and prices when available.
If there was an error, acknowledge it and suggest what the user can try.
"""


def response_generator(state: AgentState) -> AgentState:
    """Generate a natural language response for the user."""
    ds: DialogueState = state["dialogue_state"]
    ranked = state.get("ranked_results", [])
    error = state.get("error")

    # Format top results for prompt
    top_results = []
    for r in ranked[:3]:
        p = r.product
        price_str = f"${p.price:.2f}" if p.price else "N/A"
        top_results.append(f"{p.name} ({price_str}, Nutri-Score {p.nutriscore})")

    cart_total = ds.cart_total
    cart_summary = ", ".join(i.product.name or "" for i in ds.cart[:5]) or "empty"

    prompt = _RESPONSE_PROMPT.format(
        intent=ds.current_intent,
        slots=ds.slots.model_dump(),
        results="; ".join(top_results) or "none",
        cart_count=len(ds.cart),
        cart_total=f"{cart_total:.2f}" if cart_total else "0.00",
        cart_summary=cart_summary,
        error=error or "none",
    )

    try:
        response = llm.generate(prompt, role="general")
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        if error:
            response = f"I encountered an issue: {error}"
        elif ranked:
            top = ranked[0].product
            response = f"I found {len(ranked)} products. Top result: {top.name} at ${top.price:.2f}."
        else:
            response = "I'm here to help with your grocery shopping! What are you looking for?"

    ds.add_turn("assistant", response)
    return {**state, "response_text": response, "dialogue_state": ds,
            "messages": state["messages"] + [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# Checkout Agent (router target)
# ---------------------------------------------------------------------------

def checkout_handoff(state: AgentState) -> AgentState:
    """Hand the confirmed cart off to the BrowserUse checkout agent."""
    ds: DialogueState = state["dialogue_state"]
    if not ds.cart:
        return {**state, "error": "Your cart is empty. Add some items before checking out."}

    from src.browser.checkout_agent import run_checkout
    from src.llm.preference_model import update_preferences

    try:
        result = _run_async(run_checkout(
            ds.cart,
            user_id=state.get("user_id") or "default",
            scenario_id=state.get("session_id"),
        ))
    except Exception as exc:
        logger.exception("Checkout handoff failed")
        return {**state, "error": f"Checkout failed: {exc}"}

    rejected = []
    ranked = state.get("ranked_results", [])
    cart_ids = {ci.product.instacart_id for ci in ds.cart}
    for r in ranked:
        if r.product.instacart_id not in cart_ids:
            rejected.append(r.product.name)

    try:
        update_preferences(
            user_id=state.get("user_id") or "default",
            confirmed_cart=ds.cart,
            rejected_items=rejected[:5],
            new_dietary_flags=ds.dietary_preferences,
            new_budget=ds.budget,
        )
    except Exception as exc:
        logger.warning("Preference update failed: %s", exc)

    return {
        **state,
        "checkout_ready": True,
        "checkout_result": result,
    }
