"""Load SPO triples and Open Food Facts attributes into Neo4j.

Node types: Product, Ingredient, Attribute, Category
Relationship types: CONTAINS, HAS_ATTRIBUTE, IN_CATEGORY, <predicate>
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from src.knowledge_graph.spo_extractor import load_triples
from src.knowledge_graph.entity_standardizer import standardize_triples
from src.utils.config import get_settings
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

_PROCESSED = PROCESSED_DIR


def _get_driver() -> Driver:
    cfg = get_settings()
    neo4j_cfg = cfg["neo4j"]
    uri = os.getenv("NEO4J_URI", neo4j_cfg["uri"])
    user = os.getenv("NEO4J_USER", neo4j_cfg.get("user", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD", neo4j_cfg.get("password", "clickless123"))
    return GraphDatabase.driver(uri, auth=(user, password))


def _create_constraints(driver: Driver) -> None:
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.nutriscore)")
    logger.info("Neo4j constraints and indexes created")


def _batch_merge_triples(session, batch: List[dict]) -> None:
    query = """
    UNWIND $triples AS t
    MERGE (s:Entity {name: t.s})
    MERGE (o:Entity {name: t.o})
    MERGE (s)-[r:RELATES {predicate: t.p}]->(o)
    ON CREATE SET r.count = 1
    ON MATCH SET r.count = r.count + 1
    """
    session.run(query, triples=batch)


def _load_product_nodes(session, off_parquet_path: Path, batch_size: int = 500, max_products: Optional[int] = None) -> int:
    """Create Product nodes with OFF nutritional attributes."""
    import pandas as pd

    if not off_parquet_path.exists():
        logger.warning("off_enriched.parquet not found -- skipping product node creation")
        return 0

    df = pd.read_parquet(off_parquet_path)
    if max_products:
        df = df.head(max_products)
    count = 0
    batch = []

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue

        node = {
            "name": name,
            "brand": str(row.get("brand", "") or ""),
            "category": str(row.get("category", "") or ""),
            "nutriscore": str(row.get("nutriscore", "unknown") or "unknown").upper(),
            "nova_group": int(row["nova_group"]) if row.get("nova_group") and not pd.isna(row["nova_group"]) else None,
            "energy_kcal": float(row["energy_kcal"]) if row.get("energy_kcal") and not pd.isna(row["energy_kcal"]) else None,
            "protein_g": float(row["protein_g"]) if row.get("protein_g") and not pd.isna(row["protein_g"]) else None,
            "allergens": str(row.get("allergens", "") or ""),
        }
        batch.append(node)

        if len(batch) >= batch_size:
            session.run(
                """
                UNWIND $nodes AS n
                MERGE (p:Product {name: n.name})
                SET p += n
                """,
                nodes=batch,
            )
            count += len(batch)
            batch = []

    if batch:
        session.run(
            "UNWIND $nodes AS n MERGE (p:Product {name: n.name}) SET p += n",
            nodes=batch,
        )
        count += len(batch)

    return count


def load_all(
    batch_size: int = 500,
    max_triples: Optional[int] = None,
    standardize: bool = True,
    max_products: Optional[int] = None,
) -> None:
    """Main entry point: load products + triples into Neo4j."""
    driver = _get_driver()
    try:
        _create_constraints(driver)

        # Load product nodes from OFF
        with driver.session() as session:
            off_path = _PROCESSED / "off_enriched.parquet"
            n_products = _load_product_nodes(session, off_path, batch_size=batch_size, max_products=max_products)
            logger.info("Loaded %d Product nodes from OFF", n_products)

        # Load SPO triples
        raw_triples = load_triples()
        if max_triples:
            raw_triples = raw_triples[:max_triples]

        if standardize and raw_triples:
            logger.info("Standardizing %d triples...", len(raw_triples))
            triples, _ = standardize_triples(raw_triples)
        else:
            triples = raw_triples

        logger.info("Loading %d triples into Neo4j...", len(triples))
        with driver.session() as session:
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                _batch_merge_triples(session, batch)
                if (i // batch_size) % 10 == 0:
                    logger.info("  Loaded %d/%d triples", i + len(batch), len(triples))

        logger.info("Neo4j loading complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    load_all()
