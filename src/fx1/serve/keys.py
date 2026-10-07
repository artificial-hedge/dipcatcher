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
  key or lose a minted one. Accepted ``uses``/``tokens_used``/
  ``last_used_at`` snapshots are also journaled; a restart restores the
  spend. Boot compacts snapshots into the latest record per key, retaining
  revoked and quarantined records and rotation lineage.
- Every detected journal break, including a torn final line, quarantines
  recovered keys and reports ``recover_warnings``. A persisted marker
  keeps authentication required even if no complete key record survived.
  The trusted ``FX1_API_KEY`` bootstrap credential can mint replacements;
  clean later boots keep those fresh keys enabled. Unknown lost history
  is never treated as permission to reactivate a credential.
- Auth failure is uniform: bad credentials and absent credentials get
  the same 401 shape as a wrong env key — no oracle for which entries
  exist. Expired keys fail the same way — a dead credential is a dead
  credential, not a hint.
- Optional policy at mint: ``rpm`` bounds the key to a fixed 60 s
  request window (over-limit raises ``rate_limited`` with the window's
  remaining seconds — the wire maps it to 429 + ``Retry-After``) and
  ``ttl_s`` bakes an ``expires_at`` into the record. Both are journaled
  fields (declared at mint, durable policy); the live window counters
  remain process-local. A refused request — over-limit or
  expired — never bumps the use counter.
- ``scopes`` declares which surface classes the key may touch:
  ``read`` (safe methods anywhere), ``write`` (mutating calls outside
  the control plane), and ``admin`` (key management + drain). The wire
  refuses out-of-scope calls 403 ``insufficient_scope`` — a refused
  scope never counts as a use either. ``admin=True`` at mint unions the
  admin scope onto whatever ``scopes`` declares, so the flag is purely
  additive. Unset ``scopes`` keeps the pre-scope behavior:
  ``[read, write]`` (plus ``admin`` for admin keys). Scopes are
  journaled with the record — a restart restores the declared policy.
- ``max_requests`` / ``max_tokens`` declare hard budgets: the key
  refuses ``quota_exceeded`` once ``uses`` reaches ``max_requests``
  authenticated calls, or once its reported token spend (charged out of
  the completion log as providers report it) reaches ``max_tokens``.
  Budgets and their counters are durable when a journal is configured;
  a normal restart never resets acknowledged spend. A budget gates the
  *next* call, so the request crossing the token line completes before its
  provider-reported charge counts against the next call.
"""

from __future__ import annotations

import hashlib
import logging
import math
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from fx1.serve.journal import JobJournal

__all__ = ["KEY_PREFIX", "SCOPES", "ApiKeyStore", "KeyStoreError"]

_LOG = logging.getLogger(__name__)

KEY_PREFIX = "fx1k_"
_MAX_KEYS = 4096
_RATE_WINDOW_S = 60.0
# The scope vocabulary — read covers safe methods anywhere, write the
# data-plane mutations, admin the control plane (key management, drain).
SCOPES = ("read", "write", "admin")
# The nullable policy fields a PATCH may clear back to unbounded.
CLEARABLE_KEY_FIELDS = frozenset({"name", "rpm", "max_requests", "max_tokens", "expires_at"})


def _resolve_scopes(scopes: list[str] | tuple[str, ...] | None, admin: bool) -> list[str]:
    """Resolve a mint's scope declaration into the journaled list.

    Unset keeps the pre-scope contract — ``[read, write]``, plus
    ``admin`` on an admin key. ``admin=True`` unions the admin scope
    into an explicit list, never silently drops it. Empty lists and
    unknown names refuse at mint, not first use."""
    if scopes is None:
        return list(SCOPES) if admin else ["read", "write"]
    seen = {s for s in scopes}
    bad = sorted(s for s in seen if not isinstance(s, str) or s not in SCOPES)
    if bad:
        raise KeyStoreError("scopes_invalid", f"unknown key scope {bad[0]!r}")
    if not seen:
        raise KeyStoreError("scopes_empty", "scopes must name at least one of read/write/admin")
    if admin:
        seen.add("admin")
    return [s for s in SCOPES if s in seen]


def _wire(rec: dict[str, Any]) -> dict[str, Any]:
    """The public view of a record: the sha256 and any ``_``-prefixed
    live counters (rate window) never leave the store."""
    return {k: v for k, v in rec.items() if k != "sha256" and not k.startswith("_")}


def _durable_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Persist policy and lifetime counters; rate windows remain process-local."""
    return {key: value for key, value in rec.items() if not key.startswith("_")}


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _record(
    raw: str,
    sha: str,
    *,
    name: Any,
    scopes: Iterable[str],
    rpm: Any,
    max_requests: Any,
    max_tokens: Any,
    created_at: float,
    expires_at: Any,
    rotated_from: Any = None,
) -> dict[str, Any]:
    """One journaled key record — mint and rotate share the shape; the
    caller computes the policy values (scopes resolved, expiry
    inherited or fresh) and this stamps the record around them."""
    if expires_at is not None and not _is_finite_number(expires_at):
        raise ValueError("expires_at must be finite")
    resolved = list(scopes)
    return {
        "key_id": sha[:16],
        "prefix": raw[:13],
        "name": name,
        "admin": "admin" in resolved,
        "scopes": resolved,
        "rpm": rpm,
        "max_requests": max_requests,
        "max_tokens": max_tokens,
        # Token budget starts at zero; later provider-reported charges
        # are persisted as counter snapshots.
        "tokens_used": 0,
        "created_at": created_at,
        "expires_at": expires_at,
        "enabled": True,
        "revoked_at": None,
        "uses": 0,
        "last_used_at": None,
        "rotated_from": rotated_from,
        "sha256": sha,
    }


class KeyStoreError(RuntimeError):
    """Store-level refusal (cap hit, unknown key, already revoked,
    over its declared rate limit).

    Carries an HTTP-style code so the route can fail closed with the
    same shape as every other bounded surface; ``retry_after`` carries
    the remaining window seconds on ``rate_limited`` so the wire can
    set an honest ``Retry-After``. ``key_id`` names the refused
    credential on ``rate_limited`` so the wire can still answer its
    declared budget headers — it is the truncated fingerprint, never
    the secret."""

    def __init__(
        self,
        code: str,
        message: str,
        retry_after: float | None = None,
        key_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.key_id = key_id


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ApiKeyStore:
    """Hash-indexed key store. ``authenticate`` is a dict lookup on the
    sha256 — no scanning, no per-key compare-timing oracle."""

    def __init__(
        self,
        max_keys: int = _MAX_KEYS,
        journal: JobJournal | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self._lock = threading.Lock()
        self._max = max_keys
        self._by_hash: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, str] = {}
        self._journal = journal
        self._clock = clock
        self._auth_required = False
        self.recover_warnings: list[str] = []
        self.recovery_quarantined = False
        if journal is not None:
            result = journal.replay()
            self.recover_warnings = list(result.warnings)
            for payload in result.payloads:
                self._restore_payload(payload)
            # An existing empty journal can mean all credential history
            # was lost. Only an absent journal is safely unprovisioned.
            damaged = (
                result.dropped > 0
                or result.truncated_at is not None
                or (not result.payloads and journal.path.exists())
            )
            if damaged:
                # Even a torn final line may have been a confirmed revoke.
                # Keep auth enabled when no complete key record survived.
                self._auth_required = True
                self.recovery_quarantined = True
                for record in self._by_hash.values():
                    record["enabled"] = False
                    record["quarantined"] = True
                warning = (
                    f"journal {journal.path.name}: recovered keys quarantined after "
                    "unverified records; use the bootstrap credential to mint replacements"
                )
                self.recover_warnings.append(warning)
                _LOG.warning("%s", warning)
            if result.payloads or damaged:
                self._compact_locked()

    def _restore_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("auth_required") is True:
            self._auth_required = True
        if payload.get("recovery_quarantined") is True:
            self.recovery_quarantined = True
        for value in (payload.get("record"), payload.get("successor")):
            if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
                if value is not None:
                    self.recover_warnings.append(
                        "journaled key record is malformed — dropped without restoring"
                    )
                continue
            record = _durable_record(value)
            # A record that lacks ``enabled`` defaults disabled — a
            # credential is never enabled by a malformed line.
            record.setdefault("enabled", False)
            record.setdefault("admin", False)
            record.setdefault("scopes", list(SCOPES) if record["admin"] else ["read", "write"])
            record.setdefault("max_requests", None)
            record.setdefault("max_tokens", None)
            record.setdefault("tokens_used", 0)
            record.setdefault("uses", 0)
            record.setdefault("last_used_at", None)
            record.setdefault("rotated_from", None)
            if record.get("quarantined") is True:
                record["enabled"] = False
                self.recovery_quarantined = True
                self._auth_required = True
            self._by_hash[record["sha256"]] = record
            key_id = record.get("key_id")
            if isinstance(key_id, str):
                self._by_id[key_id] = record["sha256"]

    def _compact_locked(self) -> None:
        """Atomically repair/fold snapshots, retaining tombstones and auth history.

        Called during construction or while holding the key-store lock.
        The recovery marker does not quarantine fresh keys on a clean boot.
        """
        if self._journal is None:
            return
        payloads: list[dict[str, Any]] = []
        if self._auth_required:
            payloads.append(
                {"auth_required": True, "recovery_quarantined": self.recovery_quarantined}
            )
        payloads.extend({"record": _durable_record(record)} for record in self._by_hash.values())
        self._journal.compact(payloads)

    @property
    def has_keys(self) -> bool:
        """Whether any key was ever minted — a non-empty store turns on
        remote auth even without ``FX1_API_KEY`` (provisioning is the
        opt-in)."""
        return self._auth_required or bool(self._by_hash)

    def _append(self, rec: dict[str, Any]) -> None:
        if self._journal is not None:
            self._journal.append({"record": _durable_record(rec)})

    def mint(
        self,
        name: str | None = None,
        *,
        admin: bool = False,
        rpm: int | None = None,
        ttl_s: float | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
        max_requests: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Mint a key. Returns ``(raw, record)`` — the raw secret is shown
        once here and never stored.

        ``admin=True`` marks a key that may manage keys itself (mint /
        list / revoke) — the bootstrap credential mints the first admin
        key so deployments without ``FX1_API_KEY`` keep a manageable
        control plane after provisioning turns auth on.

        ``scopes`` bounds which surface classes the key may call —
        ``read``, ``write``, ``admin`` — journaled with the record so a
        restart restores the declared policy; the flag ``admin`` unions
        the admin scope in.

        ``rpm`` declares a per-key fixed-window request limit
        (refusals raise ``rate_limited``); ``ttl_s`` declares an expiry —
        both travel with the journaled record so a restart keeps the
        declared policy.

        ``max_requests`` / ``max_tokens`` declare hard lifetime budgets
        (refusals raise ``quota_exceeded``): the request budget counts
        authenticated calls; the token budget counts provider-reported
        usage charged by the wire after each served response."""
        if rpm is not None and rpm < 1:
            raise ValueError("rpm must be >= 1")
        if ttl_s is not None and (not _is_finite_number(ttl_s) or ttl_s <= 0):
            raise ValueError("ttl_s must be finite and > 0")
        if max_requests is not None and max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        resolved = _resolve_scopes(scopes, admin)
        raw = KEY_PREFIX + secrets.token_hex(20)
        sha = _hash(raw)
        created = self._clock()
        rec = _record(
            raw,
            sha,
            name=name,
            scopes=resolved,
            rpm=rpm,
            max_requests=max_requests,
            max_tokens=max_tokens,
            created_at=created,
            expires_at=(created + ttl_s) if ttl_s is not None else None,
        )
        with self._lock:
            if len(self._by_hash) >= self._max:
                raise KeyStoreError("keys_cap", f"key store is full ({self._max} keys)")
            self._append(rec)
            self._by_hash[sha] = rec
            self._by_id[rec["key_id"]] = sha
        return raw, _wire(rec)

    def _consume_window(self, rec: dict[str, Any], now: float) -> None:
        """Take one slot in the key's declared ``rpm`` window, or raise
        ``rate_limited`` when the window is exhausted. Mutates ``rec``'s
        private ``_window_*`` fields; caller holds ``self._lock``."""
        rpm = rec.get("rpm")
        if rpm is None:
            return
        start = rec.get("_window_start")
        if not isinstance(start, (int, float)) or now - start >= _RATE_WINDOW_S:
            rec["_window_start"] = now
            rec["_window_count"] = 0
        if rec["_window_count"] >= rpm:
            retry = max(0.0, _RATE_WINDOW_S - (now - rec["_window_start"]))
            raise KeyStoreError(
                "rate_limited",
                f"key exceeds its {rpm}/min request limit",
                retry_after=retry,
                key_id=rec["key_id"],
            )
        rec["_window_count"] += 1

    def authenticate(self, raw: str, *, required_scope: str | None = None) -> dict[str, Any] | None:
        """Return the wire record for a presented raw key, else None.
        Journals accepted-use counters before authorizing the request.

        Expired keys fail closed like revoked ones; a key past its
        declared ``rpm`` window raises ``rate_limited`` instead of
        answering — the wire maps that to 429. ``required_scope`` is the
        authorization bound the request needs: a key missing it raises
        ``insufficient_scope`` — the wire maps that to 403. Refusals do
        not count as uses."""
        if not isinstance(raw, str) or not raw.startswith(KEY_PREFIX):
            return None
        now = self._clock()
        sha = _hash(raw)
        with self._lock:
            rec = self._by_hash.get(sha)
            if rec is None or not rec["enabled"]:
                return None
            expires = rec.get("expires_at")
            if expires is not None and (not _is_finite_number(expires) or now >= expires):
                return None
            # hard budgets refuse before the rate window — an exhausted
            # key never consumes a window slot
            max_req = rec.get("max_requests")
            if max_req is not None and rec["uses"] >= max_req:
                raise KeyStoreError(
                    "quota_exceeded",
                    f"key exhausted its {max_req} request budget",
                    key_id=rec["key_id"],
                )
            max_tok = rec.get("max_tokens")
            if max_tok is not None and int(rec.get("tokens_used", 0)) >= max_tok:
                raise KeyStoreError(
                    "quota_exceeded",
                    f"key exhausted its {max_tok} token budget",
                    key_id=rec["key_id"],
                )
            # the scope check runs before any counter moves — a denied
            # call never counts as a use nor consumes a window slot
            if required_scope is not None and required_scope not in (rec.get("scopes") or []):
                raise KeyStoreError(
                    "insufficient_scope",
                    f"key lacks required scope {required_scope!r}",
                    key_id=rec["key_id"],
                )
            rec = dict(rec)
            self._consume_window(rec, now)
            rec["uses"] += 1
            rec["last_used_at"] = now
            self._append(rec)
            self._by_hash[sha] = rec
            return _wire(rec)

    def window_state(self, key_id: str) -> tuple[int, int, int] | None:
        """``(limit, remaining, reset_s)`` for a key's declared rpm
        window, or None when the key is unknown / has no rpm bound.

        Advisory: the count is read after ``authenticate`` consumed this
        request, so concurrent calls may shift it by the time headers are
        emitted — the wire budget is still honest at read time."""
        with self._lock:
            sha = self._by_id.get(key_id)
            rec = self._by_hash.get(sha) if sha is not None else None
            if rec is None:
                return None
            rpm = rec.get("rpm")
            if rpm is None:
                return None
            now = self._clock()
            start = rec.get("_window_start")
            if not isinstance(start, (int, float)) or now - start >= _RATE_WINDOW_S:
                return (rpm, rpm, 0)
            count = int(rec.get("_window_count", 0))
            reset = max(0, int(round(_RATE_WINDOW_S - (now - start))))
            return (rpm, max(0, rpm - count), reset)

    def charge_tokens(self, key_id: str, tokens: int) -> None:
        """Add and journal provider-reported token spend to the key's meter.
        Called by the wire after a served response — a tombstoned or
        unknown key still records the spend (audit, not auth)."""
        if tokens <= 0:
            return
        with self._lock:
            sha = self._by_id.get(key_id)
            rec = self._by_hash.get(sha) if sha is not None else None
            if rec is not None:
                # The provider has already spent these tokens. Keep the
                # live charge even when persisting it raises an I/O error.
                rec["tokens_used"] = int(rec.get("tokens_used", 0)) + int(tokens)
                self._append(rec)

    def get(self, key_id: str) -> dict[str, Any] | None:
        with self._lock:
            sha = self._by_id.get(key_id)
            rec = self._by_hash.get(sha) if sha is not None else None
            return _wire(rec) if rec else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_wire(rec) for rec in self._by_hash.values()]

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
            rec = {**rec, "enabled": False, "revoked_at": self._clock()}
            self._append(rec)
            self._by_hash[sha] = rec
            return _wire(rec)

    def rotate(
        self,
        key_id: str,
        *,
        revoke_old: bool = True,
        name: str | None = None,
        ttl_s: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Mint a successor under the predecessor's declared policy —
        same name, admin, scopes, rpm, and budgets — stamped
        ``rotated_from`` on the journaled record so the lineage survives
        a restart. ``revoke_old`` (default) tombstones the predecessor
        in the same journal line and lock: recovery sees both records
        or neither, and the old secret dies as the new one appears. With
        ``revoke_old=False`` both secrets authenticate until the old key
        is revoked or expires — the overlap is the operator's choice,
        declared on the response.

        Expiry: ``ttl_s`` mints the successor a fresh lifetime;
        omitted, it inherits the predecessor's absolute ``expires_at``
        verbatim — rotation changes the secret, never the declared
        deadline. Unknown keys raise ``key_not_found``; rotating a
        revoked credential raises ``key_revoked`` (a dead secret cannot
        mint a live one); a full store raises ``keys_cap`` BEFORE the
        predecessor is touched."""
        if ttl_s is not None and (not _is_finite_number(ttl_s) or ttl_s <= 0):
            raise ValueError("ttl_s must be finite and > 0")
        with self._lock:
            sha = self._by_id.get(key_id)
            old = self._by_hash.get(sha) if sha is not None else None
            if old is None:
                raise KeyStoreError("key_not_found", f"unknown key {key_id!r}")
            if not old.get("enabled", True):
                raise KeyStoreError("key_revoked", f"key {key_id!r} is revoked")
            if len(self._by_hash) >= self._max:
                raise KeyStoreError("keys_cap", f"key store is full ({self._max} keys)")
            raw = KEY_PREFIX + secrets.token_hex(20)
            new_sha = _hash(raw)
            created = self._clock()
            rec = _record(
                raw,
                new_sha,
                name=name if name is not None else old.get("name"),
                scopes=(
                    old.get("scopes") or (list(SCOPES) if old.get("admin") else ["read", "write"])
                ),
                rpm=old.get("rpm"),
                max_requests=old.get("max_requests"),
                max_tokens=old.get("max_tokens"),
                created_at=created,
                expires_at=(created + ttl_s) if ttl_s is not None else old.get("expires_at"),
                rotated_from=key_id,
            )
            if revoke_old and sha is not None:
                revoked = {**old, "enabled": False, "revoked_at": created}
                if self._journal is not None:
                    self._journal.append(
                        {"record": _durable_record(revoked), "successor": _durable_record(rec)}
                    )
                self._by_hash[sha] = revoked
            else:
                self._append(rec)
            self._by_hash[new_sha] = rec
            self._by_id[rec["key_id"]] = new_sha
            return raw, _wire(rec)

    def update(
        self,
        key_id: str,
        *,
        name: str | None = None,
        rpm: int | None = None,
        scopes: Iterable[str] | None = None,
        admin: bool | None = None,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        expires_at: float | None = None,
        clear: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Patch a live key's declared policy in place — no new secret,
        no slot consumed; the updated record journals like revoke's so
        a ``--state-dir`` replay reconstructs it.

        An omitted field keeps the record's value; a field named in
        ``clear`` (``name``/``rpm``/``max_requests``/``max_tokens``/
        ``expires_at``) reverts to ``None`` — the unbounded default.
        ``scopes``/``admin`` resolve the mint way: ``admin=True``
        unions the admin scope onto whichever list survives the patch
        (an explicit ``scopes`` or the record's), ``admin=False``
        never strips a scope the caller declared — the flag is purely
        additive. ``enabled`` and the live counters
        (``uses``/``tokens_used``) are not patchable — revocation is
        permanent, and ``update`` never resurrects: a tombstoned key
        raises ``key_revoked``, an unknown one ``key_not_found``."""
        if rpm is not None and rpm < 1:
            raise ValueError("rpm must be >= 1")
        if max_requests is not None and max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if expires_at is not None and (not _is_finite_number(expires_at) or expires_at <= 0):
            raise ValueError("expires_at must be finite and > 0")
        clears = set(clear)
        bad_clear = sorted(clears - CLEARABLE_KEY_FIELDS)
        if bad_clear:
            raise ValueError(f"cannot clear non-nullable field {bad_clear[0]!r}")
        with self._lock:
            sha = self._by_id.get(key_id)
            old = self._by_hash.get(sha) if sha is not None else None
            if sha is None or old is None:
                raise KeyStoreError("key_not_found", f"unknown key {key_id!r}")
            if not old.get("enabled", True):
                raise KeyStoreError("key_revoked", f"key {key_id!r} is revoked")
            rec = dict(old)
            for field in clears:
                rec[field] = None
            for field, value in (
                ("name", name),
                ("rpm", rpm),
                ("max_requests", max_requests),
                ("max_tokens", max_tokens),
                ("expires_at", expires_at),
            ):
                if value is not None:
                    rec[field] = value
            if scopes is not None or admin is not None:
                resolved = _resolve_scopes(
                    list(scopes) if scopes is not None else old.get("scopes"),
                    bool(admin),
                )
                rec["scopes"] = resolved
                rec["admin"] = "admin" in resolved
            self._append(rec)
            self._by_hash[sha] = rec
            return _wire(rec)
