"""SYNTHETIC adversarial probes for the API key store — raw secrets are
never durable, revocation is permanent, and malformed journaled records
fail closed with a warning instead of silent trust or a crash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.keys import KEY_PREFIX, ApiKeyStore, KeyStoreError


def _store(path: Path) -> ApiKeyStore:
    return ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)


def test_raw_secret_is_never_durable(tmp_path: Path) -> None:
    """The journal carries the sha256, never the raw key — and the wire
    record strips both the hash and private bookkeeping."""
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint(name="synthetic")
    journal_text = path.read_text()
    assert raw not in journal_text
    assert hashlib.sha256(raw.encode()).hexdigest() in journal_text
    assert raw not in json.dumps(wire)
    assert "sha256" not in wire
    assert not any(k.startswith("_") for k in wire)
    assert wire["prefix"] == raw[:13]  # prefix only — the secret stays out


def test_revoked_key_stays_dead_across_restarts(tmp_path: Path) -> None:
    """Revocation is a journal-first tombstone: restart after restart
    the raw key authenticates to nothing, and the record reads
    disabled."""
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint(admin=True)
    store.revoke(wire["key_id"])
    for _ in range(3):
        store = _store(path)
        assert store.authenticate(raw) is None
        rec = store.get(wire["key_id"])
        assert rec is not None and rec["enabled"] is False


def test_revoked_key_refuses_update_and_rotation(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint()
    store.revoke(wire["key_id"])
    with pytest.raises(KeyStoreError) as exc:
        store.update(wire["key_id"], rpm=10)
    assert exc.value.code == "key_revoked"
    with pytest.raises(KeyStoreError) as exc2:
        store.rotate(wire["key_id"])
    assert exc2.value.code == "key_revoked"
    assert store.authenticate(raw) is None


def test_malformed_journaled_record_is_dropped_loudly(tmp_path: Path) -> None:
    """A chained payload whose ``record`` lacks a sha256 must not
    authenticate anything — and must surface in ``recover_warnings``
    instead of passing silently."""
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, _wire_rec = store.mint(admin=True)
    journal.append({"record": {"key_id": "ghost", "prefix": "fx1k_xyz"}})  # no sha256

    recovered = _store(path)
    assert recovered.authenticate(raw) is not None  # the real key survives
    assert recovered.get("ghost") is None
    assert any("malformed" in w for w in recovered.recover_warnings)


def test_journaled_record_without_enabled_defaults_disabled(tmp_path: Path) -> None:
    """A record missing ``enabled`` must default disabled — authenticate
    returns None (never a KeyError crash that takes auth down)."""
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    store.mint(admin=True)  # provisions the store so replay isn't quarantined
    ghost_raw = KEY_PREFIX + "aa" * 20
    journal.append(
        {
            "record": {
                "key_id": "ghost1",
                "prefix": ghost_raw[:13],
                "sha256": hashlib.sha256(ghost_raw.encode()).hexdigest(),
                "uses": 0,
                "created_at": 1.0,
            }
        }
    )
    recovered = _store(path)
    assert recovered.authenticate(ghost_raw) is None
    rec = recovered.get("ghost1")
    assert rec is not None and rec["enabled"] is False


def test_rotate_is_one_atomic_transition(tmp_path: Path) -> None:
    """rotate with revoke_old journals successor + revoked predecessor
    atomically: restart never resurrects the old secret."""
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint(admin=True)
    new_raw, new_wire = store.rotate(wire["key_id"], revoke_old=True)

    recovered = _store(path)
    assert recovered.authenticate(raw) is None
    assert recovered.authenticate(new_raw) is not None
    old = recovered.get(wire["key_id"])
    new = recovered.get(new_wire["key_id"])
    assert old is not None and old["enabled"] is False
    assert new is not None and new["rotated_from"] == wire["key_id"]


def test_budgets_refuse_after_restart(tmp_path: Path) -> None:
    """Hard budgets are durable policy: an exhausted key keeps refusing
    across restarts — never a use counter reset."""
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint(max_requests=1, max_tokens=5)
    assert store.authenticate(raw) is not None
    store.charge_tokens(wire["key_id"], 5)

    recovered = _store(path)
    with pytest.raises(KeyStoreError) as exc:
        recovered.authenticate(raw)
    assert exc.value.code == "quota_exceeded"


def test_rpm_window_is_honored(tmp_path: Path) -> None:
    """The declared fixed window is honest: exceeding calls raise
    rate_limited with a positive retry_after, refusals never count."""
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, wire = store.mint(rpm=2)
    store.authenticate(raw)
    store.authenticate(raw)
    with pytest.raises(KeyStoreError) as exc:
        store.authenticate(raw)
    assert exc.value.code == "rate_limited"
    assert exc.value.retry_after is not None and exc.value.retry_after >= 0
