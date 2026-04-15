"""LangGraph StateGraph definition for ClickLess AI.

Pipeline:
  nlu_agent -> [router] -> api_agent -> kg_ranking_agent -> response_generator
                        -> cart_agent -> response_generator
                        -> checkout_handoff -> response_generator
                        -> response_generator (chit_chat / direct)
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.nlu.dialogue_state import DialogueState
from src.orchestration.agents import (
    api_agent,
    cart_agent,
    checkout_handoff,
    kg_ranking_agent,
    nlu_agent,
    response_generator,
)
from src.orchestration.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_nlu(state: AgentState) -> str:
    ds: DialogueState = state["dialogue_state"]
    intent = ds.current_intent

    if intent in ("search_product", "get_recommendation"):
        return "api_agent"
    if intent in ("add_to_cart", "remove_from_cart"):
        return "cart_agent"
    if intent == "checkout":
        return "checkout_handoff"
    # set_constraint, chit_chat, or fallback
    return "response_generator"


def _route_after_api(state: AgentState) -> str:
    if state.get("error"):
        return "response_generator"
    return "kg_ranking_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("nlu_agent", nlu_agent)
    graph.add_node("api_agent", api_agent)
    graph.add_node("kg_ranking_agent", kg_ranking_agent)
    graph.add_node("cart_agent", cart_agent)
    graph.add_node("checkout_handoff", checkout_handoff)
    graph.add_node("response_generator", response_generator)

    graph.add_edge(START, "nlu_agent")

    graph.add_conditional_edges(
        "nlu_agent",
        _route_after_nlu,
        {
            "api_agent": "api_agent",
            "cart_agent": "cart_agent",
            "checkout_handoff": "checkout_handoff",
            "response_generator": "response_generator",
        },
    )

    graph.add_conditional_edges(
        "api_agent",
        _route_after_api,
        {
            "kg_ranking_agent": "kg_ranking_agent",
            "response_generator": "response_generator",
        },
    )

    graph.add_edge("kg_ranking_agent", "response_generator")
    graph.add_edge("cart_agent", "response_generator")
    graph.add_edge("checkout_handoff", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    user_message: str,
    dialogue_state: DialogueState,
    cart=None,
    user_id: str = "default",
    session_id: str = "default",
) -> AgentState:
    """Run the full pipeline for a user message. Returns updated AgentState."""
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "dialogue_state": dialogue_state,
        "search_results": [],
        "ranked_results": [],
        "cart": cart or dialogue_state.cart,
        "checkout_ready": False,
        "checkout_result": None,
        "error": None,
        "user_id": user_id,
        "session_id": session_id,
        "response_text": None,
    }

    result = graph.invoke(initial_state)
    return result
