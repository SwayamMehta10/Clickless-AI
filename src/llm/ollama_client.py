"""Single access point for all Ollama LLM calls.

Wraps the `ollama` Python package with:
- Model routing (nlu / spo / general / vault / vision)
- Retries with exponential backoff
- Configurable timeout
- Structured JSON output helper
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import ollama
from ollama import ChatResponse, GenerateResponse

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_OLLAMA_CFG = _settings["ollama"]
_MODEL_MAP: Dict[str, str] = _OLLAMA_CFG["models"]

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


def _model_for(role: str) -> str:
    return _MODEL_MAP.get(role, _MODEL_MAP["general"])


def _retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Ollama call failed (attempt %d/%d): %s. Retrying in %.1fs",
                           attempt + 1, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Ollama call failed after {MAX_RETRIES} attempts") from last_exc


def generate(prompt: str, role: str = "general", **kwargs) -> str:
    """Single-turn text generation."""
    model = _model_for(role)
    logger.debug("generate | model=%s | prompt_len=%d", model, len(prompt))

    def _call():
        resp: GenerateResponse = ollama.generate(
            model=model,
            prompt=prompt,
            options={"num_predict": kwargs.pop("max_tokens", 512)},
            **kwargs,
        )
        return resp.response

    return _retry(_call)


def chat(messages: List[Dict[str, str]], role: str = "general", **kwargs) -> str:
    """Multi-turn chat completion."""
    model = _model_for(role)
    logger.debug("chat | model=%s | turns=%d", model, len(messages))

    def _call():
        resp: ChatResponse = ollama.chat(
            model=model,
            messages=messages,
            options={"num_predict": kwargs.pop("max_tokens", 512)},
            **kwargs,
        )
        return resp.message.content

    return _retry(_call)


def embed(text: str, role: str = "general") -> List[float]:
    """Generate embeddings for a text string."""
    model = _model_for(role)
    logger.debug("embed | model=%s | text_len=%d", model, len(text))

    def _call():
        resp = ollama.embeddings(model=model, prompt=text)
        return resp.embedding

    return _retry(_call)


def generate_json(prompt: str, role: str = "general", schema_hint: str = "") -> Any:
    """Generate and parse a JSON response. Raises ValueError if output isn't valid JSON."""
    if schema_hint:
        prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{schema_hint}"
    else:
        prompt = f"{prompt}\n\nRespond ONLY with valid JSON."

    raw = generate(prompt, role=role)
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc


def list_models() -> List[str]:
    """Return names of locally available Ollama models."""
    try:
        result = ollama.list()
        return [m.model for m in result.models]
    except Exception as exc:
        logger.error("Could not list Ollama models: %s", exc)
        return []


def is_available(role: str = "general") -> bool:
    """Check whether the model for a given role is available locally."""
    model = _model_for(role)
    available = list_models()
    return any(model in m for m in available)
