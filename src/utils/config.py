"""Load settings.yaml and .env, expose via get_settings()."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJ_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_PROJ_ROOT / ".env")


@lru_cache(maxsize=1)
def get_settings() -> dict:
    cfg_path = _PROJ_ROOT / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Allow .env overrides for sensitive values
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict) -> None:
    overrides = {
        ("instacart", "api_key"): "INSTACART_API_KEY",
        ("instacart", "base_url"): "INSTACART_BASE_URL",
        ("neo4j", "uri"): "NEO4J_URI",
        ("neo4j", "user"): "NEO4J_USER",
        ("neo4j", "password"): "NEO4J_PASSWORD",
        ("ollama", "base_url"): "OLLAMA_BASE_URL",
        ("app", "offline_catalog_mode"): "CLICKLESS_OFFLINE_CATALOG",
    }
    for (section, key), env_var in overrides.items():
        val = os.getenv(env_var)
        if val is not None:
            if section not in cfg:
                cfg[section] = {}
            # Convert string booleans
            if val.lower() in ("true", "1", "yes"):
                val = True
            elif val.lower() in ("false", "0", "no"):
                val = False
            cfg[section][key] = val
