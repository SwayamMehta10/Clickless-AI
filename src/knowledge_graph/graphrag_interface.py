"""GraphRAG interface for natural language querying of the knowledge graph.

Uses graph context + Llama 3.2 11B to answer NL queries with cited sources.
Falls back to pure Cypher-based retrieval if GraphRAG is unavailable.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.knowledge_graph.graph_query import (
    find_by_dietary_constraint,
    find_related_products,
    get_nutrition_context,
)
from src.llm import ollama_client as llm

logger = logging.getLogger(__name__)

_GRAPHRAG_PROMPT = """\
You are a nutritional knowledge assistant for a grocery shopping app.
Use the following knowledge graph context to answer the user's query.
Cite specific facts from the context. Be concise (2-4 sentences).

Knowledge Graph Context:
{context}

User Query: {query}

Answer:"""


def _build_context_for_query(
    query: str,
    dietary_flags: Optional[List[str]] = None,
    product_names: Optional[List[str]] = None,
) -> str:
    """Assemble relevant KG facts into a context string."""
    context_parts = []

    # Nutrition facts for mentioned products
    if product_names:
        for name in product_names[:3]:
            nutrition = get_nutrition_context(name)
            if nutrition:
                context_parts.append(f"[{name}] {nutrition}")

    # Dietary constraint matching
    if dietary_flags:
        matching = find_by_dietary_constraint(dietary_flags, limit=5)
        if matching:
            names = [r.get("name", "") for r in matching]
            context_parts.append(f"Products matching {dietary_flags}: {', '.join(names)}")

    # Related product suggestions
    if product_names:
        for name in product_names[:1]:
            related = find_related_products(name, top_k=3)
            if related:
                related_names = [r.get("name", "") for r in related]
                context_parts.append(f"Products related to {name}: {', '.join(related_names)}")

    return "\n".join(context_parts) if context_parts else "No relevant KG context found."


def query(
    user_query: str,
    dietary_flags: Optional[List[str]] = None,
    product_names: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Query the knowledge graph with natural language.

    Returns (answer, context_used).
    """
    context = _build_context_for_query(user_query, dietary_flags, product_names)

    prompt = _GRAPHRAG_PROMPT.format(context=context, query=user_query)
    try:
        answer = llm.generate(prompt, role="general")
        return answer.strip(), context
    except Exception as exc:
        logger.error("GraphRAG query failed: %s", exc)
        return "Unable to retrieve knowledge graph context at this time.", context


def get_relevance_score(product_name: str, query: str, dietary_flags: Optional[List[str]] = None) -> float:
    """Return a 0-1 score for how relevant a product is to the query, based on KG context."""
    try:
        nutrition = get_nutrition_context(product_name)
        if not nutrition:
            return 0.3  # Neutral score if no KG data

        prompt = f"""\
Rate the relevance of this product to the user's grocery query on a scale of 0.0 to 1.0.
Consider nutritional match, dietary constraints, and product category.

Product: {product_name}
Nutritional info: {nutrition}
User query: {query}
Dietary flags: {dietary_flags or []}

Return ONLY a JSON object: {{"score": <float 0.0 to 1.0>}}
"""
        result = llm.generate_json(prompt, role="general")
        return float(result.get("score", 0.5))
    except Exception as exc:
        logger.debug("KG relevance scoring failed for '%s': %s", product_name, exc)
        return 0.3
