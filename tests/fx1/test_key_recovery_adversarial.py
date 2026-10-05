"""SYNTHETIC independent checks of key recovery and durable admission."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore, KeyStoreError


def _store(path: Path) -> ApiKeyStore:
    return ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)


def test_existing_empty_journal_cannot_reenable_anonymous_mode(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    original = _store(path)
    secret, key = original.mint(admin=True)
    original.revoke(key["key_id"])
    path.write_bytes(b"")
    for _ in range(3):
        recovered = _store(path)
        assert recovered.has_keys is True
        assert recovered.authenticate(secret) is None
        assert recovered.recovery_quarantined is True
    fresh_secret, _ = recovered.mint(admin=True)
    for _ in range(2):
        recovered = _store(path)
        assert recovered.authenticate(secret) is None
        assert recovered.authenticate(fresh_secret, required_scope="admin") is not None


def test_scope_refusal_changes_no_durable_or_live_counter(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    secret, record = store.mint(scopes=["read"], rpm=1, max_requests=1)
    before = path.read_bytes()
    with pytest.raises(KeyStoreError) as denied:
        store.authenticate(secret, required_scope="admin")
    assert denied.value.code == "insufficient_scope"
    assert store.get(record["key_id"]) == record
    assert path.read_bytes() == before
    assert store.authenticate(secret, required_scope="read") is not None
    with pytest.raises(KeyStoreError) as exhausted:
        _store(path).authenticate(secret, required_scope="read")
    assert exhausted.value.code == "quota_exceeded"


@pytest.mark.parametrize("operation", ["authenticate", "charge", "rotate"])
def test_process_loss_after_durable_write_replays_one_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    secret, key = store.mint(max_requests=1, max_tokens=1)
    append = journal.append

    class ProcessLost(BaseException):
        pass

    def append_then_stop(payload: dict[str, Any]) -> None:
        append(payload)
        raise ProcessLost

    monkeypatch.setattr(journal, "append", append_then_stop)
    with pytest.raises(ProcessLost):
        if operation == "authenticate":
            store.authenticate(secret)
        elif operation == "charge":
            store.charge_tokens(key["key_id"], 1)
        else:
            store.rotate(key["key_id"])
    recovered = _store(path)
    if operation == "rotate":
        assert recovered.authenticate(secret) is None
        enabled = [record for record in recovered.list() if record["enabled"]]
        assert len(enabled) == 1
        assert enabled[0]["rotated_from"] == key["key_id"]
    else:
        with pytest.raises(KeyStoreError) as exhausted:
            recovered.authenticate(secret)
        assert exhausted.value.code == "quota_exceeded"


def test_quarantine_cannot_be_bypassed_by_update_or_rotation(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    secret, key = store.mint(admin=True)
    with path.open("ab") as stream:
        stream.write(b"{SYNTHETIC partial journal tail")
    for _ in range(3):
        store = _store(path)
        assert store.authenticate(secret) is None
        for operation in (
            partial(store.update, key["key_id"], admin=True, clear=["expires_at"]),
            partial(store.rotate, key["key_id"], revoke_old=False),
        ):
            with pytest.raises(KeyStoreError) as denied:
                operation()
            assert denied.value.code == "key_revoked"
