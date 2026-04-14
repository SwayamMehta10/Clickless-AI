"""Minimal Gemini REST client for demo-friendly text generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

from src.utils.config import get_settings

logger = logging.getLogger(__name__)


def _get_cfg() -> dict:
    settings = get_settings()
    gemini_cfg = settings.get("gemini", {}).copy()
    gemini_cfg["api_key"] = os.getenv("GEMINI_API_KEY", gemini_cfg.get("api_key"))
    gemini_cfg["model"] = os.getenv("GEMINI_MODEL", gemini_cfg.get("model", "gemini-2.5-flash"))
    gemini_cfg["base_url"] = os.getenv(
        "GEMINI_BASE_URL",
        gemini_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
    )
    gemini_cfg["timeout"] = int(os.getenv("GEMINI_TIMEOUT", gemini_cfg.get("timeout", 30)))
    return gemini_cfg


def is_configured() -> bool:
    return bool(_get_cfg().get("api_key"))


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def generate(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
    max_output_tokens: int = 220,
) -> str:
    """Generate text with Gemini using the REST API."""
    cfg = _get_cfg()
    api_key = cfg.get("api_key")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    url = f"{cfg['base_url']}/models/{cfg['model']}:generateContent"
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    with httpx.Client(timeout=cfg["timeout"]) as client:
        response = client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return _extract_text(data)


def generate_json(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 220,
) -> Any:
    """Generate text and parse it as JSON."""
    raw = generate(
        prompt,
        system_instruction=system_instruction,
        temperature=0.1,
        max_output_tokens=max_output_tokens,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)
