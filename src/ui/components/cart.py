"""Cart panel UI component."""

from __future__ import annotations

from typing import Callable, List, Optional

import streamlit as st

from src.api.product_schema import CartItem


def render_cart(
    cart: List[CartItem],
    on_remove: Optional[Callable[[str], None]] = None,
    on_checkout: Optional[Callable[[], None]] = None,
) -> None:
    """Render the cart panel in the sidebar."""
    st.subheader("Your Cart")

    if not cart:
        st.info("Your cart is empty.")
        return

    total = 0.0
    for item in cart:
        p = item.product
        col1, col2 = st.columns([4, 1])
        with col1:
            price_str = f"${p.price:.2f}" if p.price else "N/A"
            st.markdown(f"**{p.name}** × {item.quantity} — {price_str} ea.")
            if item.line_total:
                st.caption(f"Subtotal: ${item.line_total:.2f}")
                total += item.line_total
        with col2:
            if st.button("✕", key=f"remove_{p.instacart_id}", help="Remove item"):
                if on_remove:
                    on_remove(p.instacart_id)

    st.divider()
    st.markdown(f"**Total: ${total:.2f}**")

    if on_checkout:
        if st.button("Confirm and Checkout →", type="primary", use_container_width=True):
            on_checkout()
