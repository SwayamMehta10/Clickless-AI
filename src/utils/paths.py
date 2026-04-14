"""Project-relative filesystem paths used across the app."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREFERENCES_DIR = DATA_DIR / "preferences"
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
VAULT_PATH = DATA_DIR / "credential_store.enc"
VAULT_SALT_PATH = DATA_DIR / ".vault_salt"

