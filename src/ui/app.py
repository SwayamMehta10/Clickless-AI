"""ClickLess AI -- Streamlit main application.

Layout:
  Sidebar: zip code, budget, dietary preferences + cart panel
  Main: chat interface with product results
  Expander: knowledge graph visualization
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import streamlit as st

from src.api.product_schema import CartItem, RankedProduct
from src.nlu.dialogue_state import DialogueState
from src.orchestration.graph_builder import run_pipeline
from src.ui.components.cart import render_cart
from src.ui.components.chat import render_results
from src.ui.components.kg_viz import render_kg_panel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ClickLess AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "dialogue_state" not in st.session_state:
        st.session_state.dialogue_state = DialogueState(session_id=st.session_state.session_id)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List of {"role": str, "content": str, "results": list}
    if "ranked_results" not in st.session_state:
        st.session_state.ranked_results = []
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None
    if "checkout_triggered" not in st.session_state:
        st.session_state.checkout_triggered = False


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛒 ClickLess AI")
    st.caption("Conversational grocery shopping")
    st.divider()

    # User settings
    st.subheader("Preferences")
    zip_code = st.text_input("Zip Code", value="85281", max_chars=10)
    budget = st.number_input("Budget ($)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    dietary = st.multiselect(
        "Dietary Preferences",
        ["organic", "gluten-free", "vegan", "low-sodium", "nut-free", "dairy-free"],
        default=[],
    )

    # Apply settings to dialogue state
    ds: DialogueState = st.session_state.dialogue_state
    ds.zip_code = zip_code
    ds.budget = budget if budget > 0 else None
    ds.dietary_preferences = dietary

    st.divider()

    # Cart panel
    def _on_remove(product_id: str) -> None:
        ds.remove_from_cart(product_id)
        st.rerun()

    def _on_checkout() -> None:
        st.session_state.checkout_triggered = True
        st.rerun()

    render_cart(ds.cart, on_remove=_on_remove, on_checkout=_on_checkout)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ranked_results = []
        st.session_state.dialogue_state = DialogueState(session_id=st.session_state.session_id)
        st.rerun()

# ---------------------------------------------------------------------------
# Checkout flow
# ---------------------------------------------------------------------------

if st.session_state.checkout_triggered:
    st.session_state.checkout_triggered = False
    ds = st.session_state.dialogue_state
    if ds.cart:
        with st.status("Handing off to BrowserUse checkout...", expanded=True) as status:
            try:
                from src.browser.checkout_agent import run_checkout
                result = asyncio.run(run_checkout(
                    ds.cart,
                    user_id="default",
                    scenario_id=st.session_state.session_id,
                ))
                st.session_state.last_checkout = result
                if result.get("success"):
                    status.update(
                        label=f"Checkout complete · {result.get('items_added', 0)} items",
                        state="complete",
                    )
                    st.toast("Cart handed off to Instacart", icon="✅")
                    if result.get("live_url"):
                        st.markdown("**Live BrowserUse session:**")
                        st.components.v1.iframe(result["live_url"], height=520)
                    if result.get("cart_url"):
                        st.markdown(f"[Open Instacart cart →]({result['cart_url']})")
                    if result.get("screenshots"):
                        with st.expander(f"Step screenshots ({len(result['screenshots'])})"):
                            for shot in result["screenshots"][:12]:
                                st.image(shot, use_container_width=True)
                else:
                    status.update(label="Checkout did not complete", state="error")
                    st.error(f"Checkout failed: {result.get('error', 'Unknown error')}")
            except Exception as exc:
                status.update(label="Checkout error", state="error")
                st.error(f"Browser checkout error: {exc}")
    else:
        st.warning("Your cart is empty.")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("ClickLess AI")
st.caption("Describe what you need and I'll find the best options for you.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("results"):
            def _add_handler(rp: RankedProduct):
                item = CartItem(product=rp.product, quantity=1)
                st.session_state.dialogue_state.add_to_cart(item)
                st.session_state.selected_product = rp.product.name
                st.toast(f"Added {rp.product.name} to cart", icon="🛒")
                st.rerun()

            render_results(msg["results"], on_add=_add_handler)

# KG visualization expander
if st.session_state.selected_product:
    with st.expander(f"Knowledge Graph: {st.session_state.selected_product}", expanded=False):
        render_kg_panel(st.session_state.selected_product)

# Chat input
if prompt := st.chat_input("e.g. 'I need gluten-free bread under $5'"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                result_state = run_pipeline(
                    user_message=prompt,
                    dialogue_state=st.session_state.dialogue_state,
                    user_id="default",
                    session_id=st.session_state.session_id,
                )
                st.session_state.dialogue_state = result_state["dialogue_state"]
                response = result_state.get("response_text", "I'm here to help!")
                ranked = result_state.get("ranked_results", [])
                error = result_state.get("error")

                st.markdown(response)

                if ranked:
                    def _add_handler_live(rp: RankedProduct):
                        item = CartItem(product=rp.product, quantity=1)
                        st.session_state.dialogue_state.add_to_cart(item)
                        st.session_state.selected_product = rp.product.name
                        st.rerun()

                    render_results(ranked[:5], on_add=_add_handler_live)

                if error:
                    st.warning(f"Note: {error}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "results": ranked[:5] if ranked else [],
                })
                st.session_state.ranked_results = ranked

            except Exception as exc:
                logger.error("Pipeline error: %s", exc, exc_info=True)
                error_msg = f"Something went wrong: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
