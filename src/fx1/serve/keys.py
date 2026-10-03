"""Managed API keys — multi-tenant auth on top of the bootstrap env key.

``FX1_API_KEY`` stays the root credential: when it is set, every managed
key additionally authenticates, and only the env key (or loopback dev
mode, where no auth is required at all) may mint/revoke keys. When the
env key is unset, a non-empty store still authenticates remote clients —
provisioning a key over loopback is itself the act of enabling auth, so
a deployment can run keys-only without a root env credential.

Honesty rules:

- The raw key is returned exactly once at mint (``fx1k-…``); the store
  keeps only ``sha256(raw)`` — listing/lookups expose ``key_id`` (a
  truncated fingerprint) and ``prefix`` for display, never the hash or
  the secret.
- Revocation is a tombstone (``enabled=false``, ``revoked_at``), not a
  delete — the audit trail of which key existed stays.
- Under ``--state-dir`` every mint/revoke is journaled (hash-chained
  JSONL, verified on replay) so a restart does not resurrect a revoked
  key or lose a minted one. ``uses``/``last_used_at`` are live
  operational counters — deliberately NOT journaled (a per-request fsync
  would tax the hot path); they reset honestly to zero on restart.
- Auth failure is uniform: bad credentials and absent credentials get
  the same 401 shape as a wrong env key — no oracle for which entries
  exist.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from typing import Any

from fx1.serve.journal import JobJournal

__all__ = ["KEY_PREFIX", "ApiKeyStore", "KeyStoreError"]

KEY_PREFIX = "fx1k_"
_MAX_KEYS = 4096


class KeyStoreError(RuntimeError):
    """Store-level refusal (cap hit, unknown key, already revoked).

    Carries an HTTP-style code so the route can fail closed with the
    same shape as every other bounded surface."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ApiKeyStore:
    """Hash-indexed key store. ``authenticate`` is a dict lookup on the
    sha256 — no scanning, no per-key compare-timing oracle."""

    def __init__(self, max_keys: int = _MAX_KEYS, journal: JobJournal | None = None) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._lock = threading.Lock()
        self._max = max_keys
        self._by_hash: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, str] = {}  # key_id -> sha256
        self._journal = journal
        if journal is not None:
            res = journal.replay()
            for payload in res.payloads:
                rec = payload.get("record")
                if isinstance(rec, dict) and isinstance(rec.get("sha256"), str):
                    rec.setdefault("admin", False)
                    self._by_hash[rec["sha256"]] = rec
                    kid = rec.get("key_id")
                    if isinstance(kid, str):
                        self._by_id[kid] = rec["sha256"]

    @property
    def has_keys(self) -> bool:
        """Whether any key was ever minted — a non-empty store turns on
        remote auth even without ``FX1_API_KEY`` (provisioning is the
        opt-in)."""
        return bool(self._by_id)

    def _append(self, rec: dict[str, Any]) -> None:
        if self._journal is not None:
            self._journal.append({"record": rec})

    def mint(self, name: str | None = None, *, admin: bool = False) -> tuple[str, dict[str, Any]]:
        """Mint a key. Returns ``(raw, record)`` — the raw secret is shown
        once here and never stored.

        ``admin=True`` marks a key that may manage keys itself (mint /
        list / revoke) — the bootstrap credential mints the first admin
        key so deployments without ``FX1_API_KEY`` keep a manageable
        control plane after provisioning turns auth on."""
        raw = KEY_PREFIX + secrets.token_hex(20)
        sha = _hash(raw)
        rec: dict[str, Any] = {
            "key_id": sha[:16],
            "prefix": raw[:13],
            "name": name,
            "admin": admin,
            "created_at": time.time(),
            "enabled": True,
            "revoked_at": None,
            "uses": 0,
            "last_used_at": None,
            "sha256": sha,
        }
        with self._lock:
            if len(self._by_hash) >= self._max:
                raise KeyStoreError("keys_cap", f"key store is full ({self._max} keys)")
            self._append(rec)
            self._by_hash[sha] = rec
            self._by_id[rec["key_id"]] = sha
        wire = {k: v for k, v in rec.items() if k != "sha256"}
        return raw, wire

    def authenticate(self, raw: str) -> dict[str, Any] | None:
        """Return the wire record for a presented raw key, else None.
        Bumps the live use counters (not journaled)."""
        if not isinstance(raw, str) or not raw.startswith(KEY_PREFIX):
            return None
        with self._lock:
            rec = self._by_hash.get(_hash(raw))
            if rec is None or not rec["enabled"]:
                return None
            rec["uses"] += 1
            rec["last_used_at"] = time.time()
            return {k: v for k, v in rec.items() if k != "sha256"}

    def get(self, key_id: str) -> dict[str, Any] | None:
        with self._lock:
            sha = self._by_id.get(key_id)
            rec = self._by_hash.get(sha) if sha is not None else None
            return {k: v for k, v in rec.items() if k != "sha256"} if rec else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {k: v for k, v in rec.items() if k != "sha256"} for rec in self._by_hash.values()
            ]

    def revoke(self, key_id: str) -> dict[str, Any]:
        """Tombstone the key — the record stays for audit, ``enabled``
        flips false and ``revoked_at`` is stamped. Idempotent-refusing:
        revoking an already-revoked or unknown key raises."""
        with self._lock:
            sha = self._by_id.get(key_id)
            if sha is None:
                raise KeyStoreError("key_not_found", f"unknown key {key_id!r}")
            rec = self._by_hash[sha]
            if not rec["enabled"]:
                raise KeyStoreError("key_revoked", f"key {key_id!r} is already revoked")
            rec = {**rec, "enabled": False, "revoked_at": time.time()}
            self._append(rec)
            self._by_hash[sha] = rec
            return {k: v for k, v in rec.items() if k != "sha256"}
