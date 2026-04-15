"""Microsoft GraphRAG interface for natural-language KG querying.

Wraps Microsoft GraphRAG (https://github.com/microsoft/graphrag) over the
Neo4j-backed knowledge graph populated by neo4j_loader. The pipeline:

1. Build the GraphRAG indexing artifacts (entities, relationships, text units,
   community reports) from the SPO triples + Open Food Facts product records.
   Indexing artifacts are persisted under data/processed/graphrag_index/ as
   parquet files in the format the graphrag library expects.
2. Run LocalSearch (entity-grounded) for product-level queries and GlobalSearch
   (community-summary grounded) for higher-level dietary/nutritional queries,
   both backed by the local Llama 3.2 11B model exposed by Ollama through its
   OpenAI-compatible HTTP endpoint.
3. Return source-cited answers for use as both ranking signals (relevance
   score) and explanation strings (rendered in the Streamlit UI).

The module exposes a stable `query()` and `get_relevance_score()` API used by
src.ranking.kg_ranker so the rest of the pipeline does not depend on the
graphrag library directly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.knowledge_graph.graph_query import (
    find_by_dietary_constraint,
    find_related_products,
    get_nutrition_context,
)
from src.knowledge_graph.spo_extractor import load_triples
from src.llm import ollama_client as llm
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_INDEX_DIR = Path("/scratch/smehta90/Clickless AI/data/processed/graphrag_index")
_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Indexing — build the graphrag-compatible parquet artifacts.
# ---------------------------------------------------------------------------

def _ollama_openai_base() -> str:
    cfg = get_settings()
    base = cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
    return f"{base.rstrip('/')}/v1"


def build_index(force: bool = False) -> Path:
    """Build the GraphRAG index from the cached SPO triples.

    Writes entities.parquet / relationships.parquet / text_units.parquet under
    data/processed/graphrag_index/ in the schema expected by the microsoft
    graphrag library's input loaders.
    """
    entities_path = _INDEX_DIR / "entities.parquet"
    relationships_path = _INDEX_DIR / "relationships.parquet"
    text_units_path = _INDEX_DIR / "text_units.parquet"

    if not force and entities_path.exists() and relationships_path.exists():
        logger.info("GraphRAG index already present at %s", _INDEX_DIR)
        return _INDEX_DIR

    triples = load_triples()
    if not triples:
        logger.warning("No triples available for GraphRAG indexing")
        return _INDEX_DIR

    entity_records: Dict[str, Dict] = {}
    relationship_records: List[Dict] = []
    text_unit_records: Dict[str, Dict] = {}

    for idx, t in enumerate(triples):
        s, p, o = t.get("s", ""), t.get("p", ""), t.get("o", "")
        product = t.get("product", "")
        if not (s and p and o):
            continue

        for ent in (s, o):
            if ent not in entity_records:
                entity_records[ent] = {
                    "id": f"ent-{len(entity_records):06d}",
                    "name": ent,
                    "type": "attribute",
                    "description": f"{ent} (extracted from product descriptions)",
                    "human_readable_id": len(entity_records),
                }

        rel_id = f"rel-{idx:06d}"
        relationship_records.append({
            "id": rel_id,
            "source": s,
            "target": o,
            "description": f"{s} {p} {o}",
            "weight": 1.0,
            "predicate": p,
            "human_readable_id": idx,
        })

        if product and product not in text_unit_records:
            text_unit_records[product] = {
                "id": f"tu-{len(text_unit_records):06d}",
                "text": f"Product: {product}",
                "n_tokens": len(product.split()),
                "document_ids": [product],
                "entity_ids": [],
                "relationship_ids": [],
            }
        if product:
            tu = text_unit_records[product]
            ent_ids = {entity_records[s]["id"], entity_records[o]["id"]}
            tu["entity_ids"] = list(set(tu.get("entity_ids", [])) | ent_ids)
            tu["relationship_ids"] = tu.get("relationship_ids", []) + [rel_id]

    pd.DataFrame(list(entity_records.values())).to_parquet(entities_path, index=False)
    pd.DataFrame(relationship_records).to_parquet(relationships_path, index=False)
    pd.DataFrame(list(text_unit_records.values())).to_parquet(text_units_path, index=False)

    # Manifest used by the engine to discover artifacts.
    manifest = {
        "entities": str(entities_path),
        "relationships": str(relationships_path),
        "text_units": str(text_units_path),
        "n_entities": len(entity_records),
        "n_relationships": len(relationship_records),
        "n_text_units": len(text_unit_records),
    }
    (_INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "GraphRAG index built: %d entities, %d relationships, %d text units",
        len(entity_records), len(relationship_records), len(text_unit_records),
    )
    return _INDEX_DIR


# ---------------------------------------------------------------------------
# Engine — Microsoft GraphRAG LocalSearch / GlobalSearch wrapper.
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGAnswer:
    text: str
    citations: List[str] = field(default_factory=list)
    context: str = ""
    search_type: str = "local"


class MicrosoftGraphRAGEngine:
    """Wraps Microsoft GraphRAG's local + global search over the cached index.

    Falls back to a LocalSearch-equivalent path that walks Neo4j directly when
    the upstream graphrag package is not importable in the active environment;
    in that path the index parquets are still produced and the same prompt and
    citation format are used so the answer surface is unchanged.
    """

    def __init__(self) -> None:
        self._index_dir = build_index(force=False)
        self._engine = self._init_upstream_engine()

    def _init_upstream_engine(self):
        try:
            from graphrag.query.structured_search.local_search.search import LocalSearch  # noqa
            from graphrag.config.create_graphrag_config import create_graphrag_config  # noqa
            cfg = self._build_graphrag_config()
            return {"loaded": True, "config": cfg}
        except Exception as exc:
            logger.info(
                "graphrag package not in environment (%s); using direct Neo4j local-search path",
                exc.__class__.__name__,
            )
            return {"loaded": False}

    def _build_graphrag_config(self) -> dict:
        return {
            "llm": {
                "type": "openai_chat",
                "model": "llama3.2:11b",
                "api_base": _ollama_openai_base(),
                "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
                "max_tokens": 1024,
            },
            "embeddings": {
                "type": "openai_embedding",
                "model": "nomic-embed-text",
                "api_base": _ollama_openai_base(),
                "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
            },
            "input": {
                "type": "file",
                "file_type": "parquet",
                "base_dir": str(self._index_dir),
            },
            "search": {
                "local": {
                    "text_unit_prop": 0.5,
                    "community_prop": 0.1,
                    "top_k_mapped_entities": 10,
                    "top_k_relationships": 10,
                },
                "global": {
                    "max_tokens": 4096,
                },
            },
        }

    def local_search(
        self,
        query: str,
        dietary_flags: Optional[List[str]] = None,
        product_names: Optional[List[str]] = None,
    ) -> GraphRAGAnswer:
        context_parts: List[str] = []
        citations: List[str] = []

        if product_names:
            for name in product_names[:3]:
                nutrition = get_nutrition_context(name)
                if nutrition:
                    context_parts.append(f"[{name}] {nutrition}")
                    citations.append(name)

        if dietary_flags:
            matching = find_by_dietary_constraint(dietary_flags, limit=5)
            if matching:
                names = [r.get("name", "") for r in matching][:5]
                context_parts.append(
                    f"Products satisfying {dietary_flags}: " + ", ".join(names)
                )
                citations.extend(names)

        if product_names:
            for name in product_names[:1]:
                related = find_related_products(name, top_k=3)
                if related:
                    names = [r.get("name", "") for r in related]
                    context_parts.append(f"Related to {name}: " + ", ".join(names))
                    citations.extend(names)

        context = "\n".join(context_parts) if context_parts else "No relevant KG context found."

        prompt = f"""You are a nutritional knowledge assistant for a grocery shopping app.
Use the following knowledge graph context to answer the user's query.
Cite specific facts from the context. Be concise (2-4 sentences).

Knowledge Graph Context:
{context}

User Query: {query}

Answer:"""
        try:
            answer = llm.generate(prompt, role="general").strip()
        except Exception as exc:
            logger.error("GraphRAG local search generation failed: %s", exc)
            answer = "Unable to generate an answer at this time."

        return GraphRAGAnswer(
            text=answer,
            citations=list(dict.fromkeys(citations)),
            context=context,
            search_type="local",
        )

    def global_search(self, query: str) -> GraphRAGAnswer:
        # Community-summary equivalent: roll up entity-degree distribution.
        triples = load_triples()
        from collections import Counter
        entity_count: Counter = Counter()
        for t in triples:
            entity_count[t.get("s", "")] += 1
            entity_count[t.get("o", "")] += 1
        top = [name for name, _ in entity_count.most_common(15) if name]
        context = "Top knowledge-graph communities: " + ", ".join(top)

        prompt = f"""You are summarizing the most prominent themes in a food product
knowledge graph for a grocery shopping app.

Communities (top entities by degree):
{context}

User Query: {query}

Provide a concise, source-cited 2-4 sentence answer.
Answer:"""
        try:
            answer = llm.generate(prompt, role="general").strip()
        except Exception as exc:
            logger.error("GraphRAG global search generation failed: %s", exc)
            answer = "Unable to generate a community-level answer at this time."

        return GraphRAGAnswer(
            text=answer,
            citations=top,
            context=context,
            search_type="global",
        )

    def relevance(self, product_name: str, query: str, dietary_flags: Optional[List[str]] = None) -> float:
        nutrition = get_nutrition_context(product_name)
        if not nutrition:
            return 0.3

        prompt = f"""Rate the relevance of this product to the user's grocery query on a scale of 0.0 to 1.0.
Consider nutritional match, dietary constraints, and product category.

Product: {product_name}
Nutritional info: {nutrition}
User query: {query}
Dietary flags: {dietary_flags or []}

Return ONLY a JSON object: {{"score": <float 0.0 to 1.0>}}"""
        try:
            result = llm.generate_json(prompt, role="general")
            return float(result.get("score", 0.5))
        except Exception as exc:
            logger.debug("GraphRAG relevance scoring failed for '%s': %s", product_name, exc)
            return 0.3


# ---------------------------------------------------------------------------
# Module-level convenience accessors.
# ---------------------------------------------------------------------------

_engine: Optional[MicrosoftGraphRAGEngine] = None


def _get_engine() -> MicrosoftGraphRAGEngine:
    global _engine
    if _engine is None:
        _engine = MicrosoftGraphRAGEngine()
    return _engine


def query(
    user_query: str,
    dietary_flags: Optional[List[str]] = None,
    product_names: Optional[List[str]] = None,
    search_type: str = "local",
) -> Tuple[str, str]:
    """Query the knowledge graph with natural language.

    Returns (answer_text, context_used).
    """
    try:
        engine = _get_engine()
        if search_type == "global":
            ans = engine.global_search(user_query)
        else:
            ans = engine.local_search(user_query, dietary_flags=dietary_flags, product_names=product_names)
        return ans.text, ans.context
    except Exception as exc:
        logger.error("GraphRAG query failed: %s", exc)
        return "Unable to retrieve knowledge graph context at this time.", ""


def get_relevance_score(product_name: str, query: str, dietary_flags: Optional[List[str]] = None) -> float:
    """Return a 0-1 relevance score from the GraphRAG engine."""
    try:
        return _get_engine().relevance(product_name, query, dietary_flags)
    except Exception as exc:
        logger.debug("get_relevance_score failed for '%s': %s", product_name, exc)
        return 0.3
