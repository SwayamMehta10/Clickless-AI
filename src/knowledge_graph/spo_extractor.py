"""Extract Subject-Predicate-Object triples from Open Food Facts text using Mistral 7B.

Chunks OFF product descriptions, extracts triples, saves to triples.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, List, Tuple

import pandas as pd

from src.llm import ollama_client as llm

logger = logging.getLogger(__name__)

_PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")
_TRIPLES_PATH = _PROCESSED / "triples.jsonl"

_EXTRACT_PROMPT = """\
Extract factual Subject-Predicate-Object triples from this food product text.
Rules:
- Predicate must be 1-3 words (e.g. "contains", "is made from", "has", "provides")
- Subject and Object should be specific (product name, ingredient, nutrient, attribute)
- Only include verifiable facts from the text, no inferences
- Return ONLY a JSON array of [subject, predicate, object] triples

Text: {text}

Triples (JSON array):"""


def _chunk_text(text: str, chunk_size: int = 250, overlap: float = 0.10) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    step = max(1, int(chunk_size * (1 - overlap)))
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def extract_triples_from_text(product_name: str, text: str) -> List[Tuple[str, str, str]]:
    """Extract SPO triples from a product's text description."""
    if not text or len(text.split()) < 5:
        return []

    chunks = _chunk_text(text)
    all_triples = []

    for chunk in chunks:
        prompt = _EXTRACT_PROMPT.format(text=chunk)
        try:
            result = llm.generate_json(prompt=prompt, role="spo")
            if isinstance(result, list):
                for triple in result:
                    # Accept either [s, p, o] array form or {subject, predicate, object} dict form
                    if isinstance(triple, (list, tuple)) and len(triple) == 3:
                        s, p, o = [str(x).strip() for x in triple]
                    elif isinstance(triple, dict):
                        s = str(triple.get("subject") or triple.get("s") or "").strip()
                        p = str(triple.get("predicate") or triple.get("p") or "").strip()
                        o = str(triple.get("object") or triple.get("o") or "").strip()
                    else:
                        continue
                    if s and p and o:
                        all_triples.append((s, p, o))
        except ValueError as exc:
            logger.debug("Triple extraction failed for chunk: %s", exc)

    return all_triples


def extract_from_off_dataset(
    max_products: int = 10_000,
    resume: bool = True,
) -> None:
    """Batch-extract triples from OFF dataset and write to triples.jsonl."""
    path = _PROCESSED / "off_enriched.parquet"
    if not path.exists():
        raise FileNotFoundError("off_enriched.parquet not found. Run preprocess_off.py first.")

    df = pd.read_parquet(path, columns=["name", "ingredients", "category"])
    df = df[df["ingredients"].notna()].head(max_products)

    # Resume: skip already-processed products
    processed_names = set()
    if resume and _TRIPLES_PATH.exists():
        with open(_TRIPLES_PATH) as f:
            for line in f:
                obj = json.loads(line)
                processed_names.add(obj.get("product"))
        logger.info("Resuming: skipping %d already-processed products", len(processed_names))

    count = 0
    with open(_TRIPLES_PATH, "a") as out:
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            if name in processed_names:
                continue

            ingredients_text = str(row.get("ingredients", ""))
            triples = extract_triples_from_text(name, ingredients_text)

            for s, p, o in triples:
                out.write(json.dumps({"product": name, "s": s, "p": p, "o": o}) + "\n")

            count += 1
            if count % 100 == 0:
                logger.info("Processed %d/%d products", count, len(df))

    logger.info("Triple extraction complete. Total products processed: %d", count)


def load_triples() -> List[dict]:
    """Load all extracted triples from disk."""
    if not _TRIPLES_PATH.exists():
        return []
    triples = []
    with open(_TRIPLES_PATH) as f:
        for line in f:
            triples.append(json.loads(line.strip()))
    return triples
