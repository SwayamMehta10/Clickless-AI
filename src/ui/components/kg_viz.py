"""Knowledge graph visualization using PyVis, embedded via st.components.v1.html."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


def render_kg_subgraph(subgraph: Dict, height: int = 500) -> None:
    """Render a product knowledge subgraph using PyVis."""
    if not subgraph or not subgraph.get("nodes"):
        st.info("No knowledge graph data available for this product.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("PyVis not installed. Cannot render knowledge graph.")
        return

    net = Network(height=f"{height}px", width="100%", bgcolor="#0e1117", font_color="white")
    net.set_options("""
    {
      "nodes": {"font": {"color": "white"}},
      "edges": {"font": {"color": "#aaaaaa", "size": 10}},
      "physics": {"stabilization": {"iterations": 100}}
    }
    """)

    # Add nodes
    label_colors = {
        "Product": "#1f78b4",
        "Ingredient": "#33a02c",
        "Attribute": "#e31a1c",
        "Category": "#ff7f00",
        "Entity": "#6a3d9a",
    }
    for node in subgraph["nodes"]:
        color = label_colors.get(node.get("label", "Entity"), "#888888")
        net.add_node(
            node["id"],
            label=node.get("name", node["id"])[:30],
            color=color,
            title=f"{node.get('label', 'Entity')}: {node.get('name', '')}",
        )

    # Add edges
    for edge in subgraph["edges"]:
        net.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("predicate", "")[:20],
            title=edge.get("predicate", ""),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_path = f.name

    html_content = Path(html_path).read_text()
    components.html(html_content, height=height + 50, scrolling=False)


def render_kg_panel(product_name: Optional[str] = None) -> None:
    """Render the KG visualization panel, fetching subgraph for a product."""
    st.subheader("Knowledge Graph")

    if not product_name:
        st.info("Select a product to explore its knowledge graph connections.")
        return

    with st.spinner(f"Loading KG for '{product_name}'..."):
        try:
            from src.knowledge_graph.graph_query import get_product_subgraph
            subgraph = get_product_subgraph(product_name, depth=2)
            node_count = len(subgraph.get("nodes", []))
            edge_count = len(subgraph.get("edges", []))
            st.caption(f"{node_count} nodes | {edge_count} relationships")
            render_kg_subgraph(subgraph)
        except Exception as exc:
            logger.warning("KG visualization failed: %s", exc)
            st.warning(f"Knowledge graph unavailable: {exc}")
