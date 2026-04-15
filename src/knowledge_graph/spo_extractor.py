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
    """Batch-extract triples from OFF dataset and write to triples.jsonl.

    Each line records the source product name AND the OFF product code so the
    Neo4j loader can attach extracted entities back to their source Product
    node via (:Product)-[:HAS_ATTRIBUTE]->(:Entity) edges.
    """
    path = _PROCESSED / "off_enriched.parquet"
    if not path.exists():
        raise FileNotFoundError("off_enriched.parquet not found. Run preprocess_off.py first.")

    cols = ["name", "ingredients", "category"]
    has_code = False
    code_col = None
    for candidate in ("barcode", "code", "product_code"):
        try:
            df = pd.read_parquet(path, columns=cols + [candidate])
            has_code = True
            code_col = candidate
            break
        except Exception:
            continue
    if not has_code:
        df = pd.read_parquet(path, columns=cols)

    df = df[df["ingredients"].notna()].head(max_products)

    # Resume state: track by BOTH barcode and name so rows written by older
    # runs (which didn't record barcodes) still match the current
    # code-key-based iteration.
    processed_codes: set = set()
    processed_names: set = set()
    if resume and _TRIPLES_PATH.exists():
        with open(_TRIPLES_PATH) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = obj.get("product_code")
                n = obj.get("product")
                if c:
                    processed_codes.add(c)
                if n:
                    processed_names.add(n)
        logger.info(
            "Resuming: %d codes + %d names already processed",
            len(processed_codes), len(processed_names),
        )

    count = 0
    skipped_in_loop = 0
    with open(_TRIPLES_PATH, "a") as out:
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            code = str(row.get(code_col, "")) if has_code else ""
            if (code and code in processed_codes) or (name and name in processed_names):
                skipped_in_loop += 1
                continue

            ingredients_text = str(row.get("ingredients", ""))
            triples = extract_triples_from_text(name, ingredients_text)

            if triples:
                for s, p, o in triples:
                    rec = {"product": name, "product_code": code, "s": s, "p": p, "o": o}
                    out.write(json.dumps(rec) + "\n")
            else:
                # Sentinel marker: products that produced zero triples must
                # still be recorded so resume() can skip them next run.
                marker = {"product": name, "product_code": code, "s": "", "p": "", "o": ""}
                out.write(json.dumps(marker) + "\n")

            out.flush()
            if code:
                processed_codes.add(code)
            if name:
                processed_names.add(name)
            count += 1
            if count % 100 == 0:
                logger.info(
                    "Processed %d new this run / %d skipped / %d total df rows",
                    count, skipped_in_loop, len(df),
                )

    logger.info("Triple extraction complete. Total products processed: %d", count)


def load_triples() -> List[dict]:
    """Load all extracted triples from disk, filtering out resume markers."""
    if not _TRIPLES_PATH.exists():
        return []
    triples = []
    with open(_TRIPLES_PATH) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("s") and obj.get("p") and obj.get("o"):
                triples.append(obj)
    return triples
