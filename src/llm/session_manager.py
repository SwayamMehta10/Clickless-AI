"""Session token management using Llama 3.2 11B.

Generates and validates ephemeral session tokens for Instacart browser handoff.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Optional

from src.llm.credential_vault import get_vault

logger = logging.getLogger(__name__)

_TOKEN_TTL = 3600  # seconds


class SessionManager:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._vault = get_vault()

    def create_session(self) -> str:
        """Generate a new session token and store it in the vault."""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + _TOKEN_TTL
        self._vault.set(f"session:{self.user_id}", {"token": token, "expiry": expiry})
        logger.info("Created session for user %s (expires in %ds)", self.user_id, _TOKEN_TTL)
        return token

    def get_session(self) -> Optional[str]:
        """Return the active session token if valid, else None."""
        record = self._vault.get(f"session:{self.user_id}")
        if not record:
            return None
        if time.time() > record["expiry"]:
            logger.info("Session expired for user %s", self.user_id)
            self.invalidate_session()
            return None
        return record["token"]

    def invalidate_session(self) -> None:
        self._vault.delete(f"session:{self.user_id}")

    def get_or_create(self) -> str:
        token = self.get_session()
        if token is None:
            token = self.create_session()
        return token
