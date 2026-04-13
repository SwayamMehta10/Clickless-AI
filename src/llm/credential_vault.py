"""Encrypted credential vault using Fernet symmetric encryption.

Stores credentials (e.g. Instacart session tokens) in an encrypted file.
PBKDF2 key derivation from a master password.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

_VAULT_PATH = Path("/scratch/smehta90/Clickless AI/data/credential_store.enc")
_SALT_PATH = Path("/scratch/smehta90/Clickless AI/data/.vault_salt")


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def _get_salt() -> bytes:
    if _SALT_PATH.exists():
        return _SALT_PATH.read_bytes()
    salt = os.urandom(16)
    _SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SALT_PATH.write_bytes(salt)
    return salt


class CredentialVault:
    """Manage encrypted storage of credentials."""

    def __init__(self, master_password: Optional[str] = None) -> None:
        self._password = master_password or os.getenv("VAULT_MASTER_PASSWORD", "changeme_before_use")
        salt = _get_salt()
        key = _derive_key(self._password, salt)
        self._fernet = Fernet(key)
        self._cache: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if not _VAULT_PATH.exists():
            return {}
        try:
            encrypted = _VAULT_PATH.read_bytes()
            plaintext = self._fernet.decrypt(encrypted)
            self._cache = json.loads(plaintext)
            return self._cache
        except Exception as exc:
            logger.error("Failed to decrypt vault: %s", exc)
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        plaintext = json.dumps(data).encode()
        encrypted = self._fernet.encrypt(plaintext)
        _VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VAULT_PATH.write_bytes(encrypted)
        self._cache = data

    def set(self, key: str, value: Any) -> None:
        """Store a credential."""
        data = self._load()
        data[key] = value
        self._save(data)
        logger.debug("Stored credential: %s", key)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a credential."""
        return self._load().get(key, default)

    def delete(self, key: str) -> None:
        data = self._load()
        data.pop(key, None)
        self._save(data)

    def list_keys(self) -> list:
        return list(self._load().keys())


# Module-level singleton for convenience
_vault: Optional[CredentialVault] = None


def get_vault() -> CredentialVault:
    global _vault
    if _vault is None:
        _vault = CredentialVault()
    return _vault
