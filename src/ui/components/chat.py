"""Chat UI components -- product cards and message rendering."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from src.api.product_schema import RankedProduct


def nutriscore_badge(score: str) -> str:
    """Return an HTML badge for Nutri-Score."""
    colors = {
        "A": "#038141", "B": "#85BB2F", "C": "#FECB02",
        "D": "#EE8100", "E": "#E63E11", "unknown": "#888888",
    }
    color = colors.get(str(score).upper(), "#888888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em;">NS {score}</span>'
    )


def nova_badge(group: Optional[int]) -> str:
    if group is None:
        return ""
    colors = {1: "#038141", 2: "#85BB2F", 3: "#EE8100", 4: "#E63E11"}
    color = colors.get(group, "#888888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;">NOVA {group}</span>'
    )


def product_card(rp: RankedProduct, index: int, on_add=None) -> None:
    """Render a product card with image, price, nutriscore, and add-to-cart button."""
    p = rp.product
    with st.container(border=True):
        col_img, col_info, col_action = st.columns([1, 4, 2])

        with col_img:
            if p.image_url:
                st.image(p.image_url, width=70)
            else:
                st.markdown("🛒", unsafe_allow_html=False)

        with col_info:
            rank_label = f"#{rp.rank}" if rp.rank else ""
            name_display = f"**{rank_label} {p.name or 'Unknown Product'}**"
            st.markdown(name_display)

            if p.brand:
                st.caption(p.brand)

            price_str = f"${p.price:.2f}" if p.price else "Price unavailable"
            badges = nutriscore_badge(str(p.nutriscore))
            if p.nova_group:
                badges += " " + nova_badge(p.nova_group)

            st.markdown(
                f"{price_str} &nbsp; {badges}",
                unsafe_allow_html=True,
            )

            if p.allergens:
                st.caption(f"Allergens: {', '.join(p.allergens[:3])}")

            if rp.copurchase_suggestions:
                st.caption(f"Often bought with: {', '.join(rp.copurchase_suggestions[:2])}")

            # Score breakdown expander
            with st.expander("Score details", expanded=False):
                for k, v in rp.score_breakdown.items():
                    st.progress(min(float(v), 1.0), text=f"{k}: {v:.3f}")

        with col_action:
            avail_label = "✓ In stock" if p.availability else "✗ Out of stock"
            st.caption(avail_label)
            if st.button("Add to cart", key=f"add_{p.instacart_id}_{index}", disabled=not p.availability):
                if on_add:
                    on_add(rp)


def render_results(ranked: List[RankedProduct], on_add=None) -> None:
    """Render a list of ranked products."""
    if not ranked:
        st.info("No products found. Try rephrasing your query.")
        return
    st.markdown(f"**Found {len(ranked)} products:**")
    for i, rp in enumerate(ranked):
        product_card(rp, i, on_add=on_add)
