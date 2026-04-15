"""LangGraph AgentState definition."""

from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.api.product_schema import CartItem, Product, RankedProduct
from src.nlu.dialogue_state import DialogueState


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    dialogue_state: DialogueState
    search_results: List[Product]
    ranked_results: List[RankedProduct]
    cart: List[CartItem]
    checkout_ready: bool
    checkout_result: Optional[dict]
    error: Optional[str]
    # Metadata
    user_id: Optional[str]
    session_id: Optional[str]
    response_text: Optional[str]
