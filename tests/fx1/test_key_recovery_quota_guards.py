"""Journal damage cannot restore credentials or reset acknowledged budgets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore, KeyStoreError


def _store(path: Path) -> ApiKeyStore:
    return ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)


def _damage_line(path: Path, index: int, *, torn: bool = False) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    if torn:
        lines[index] = lines[index][: len(lines[index]) // 2]
    else:
        line = json.loads(lines[index])
        line["sha256"] = "0" * 64
        lines[index] = (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(b"".join(lines))


@pytest.mark.parametrize("damage", ["final_hash", "final_torn", "middle_hash"])
def test_corrupted_revocation_quarantines_history_and_allows_fresh_replacements(
    tmp_path: Path, damage: str
) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    old_secret, old = store.mint(name="original-admin", admin=True)
    store.revoke(old["key_id"])
    if damage == "middle_hash":
        store.mint(name="lost-unverified-suffix")
    _damage_line(path, 1, torn=damage == "final_torn")

    recovered = _store(path)
    assert recovered.authenticate(old_secret) is None
    assert recovered.recover_warnings
    assert recovered.recovery_quarantined is True
    assert recovered.has_keys is True
    retained = recovered.get(old["key_id"])
    assert retained is not None
    assert retained["name"] == "original-admin"
    assert retained["enabled"] is False
    assert retained["quarantined"] is True
    with pytest.raises(KeyStoreError) as blocked:
        recovered.update(old["key_id"], name="cannot-reactivate")
    assert blocked.value.code == "key_revoked"
    # This is the primitive the trusted bootstrap administrator invokes.
    replacement, fresh = recovered.mint(name="replacement-admin", admin=True)
    for _ in range(2):
        recovered = _store(path)
        assert recovered.authenticate(old_secret) is None
        assert recovered.authenticate(replacement, required_scope="admin") is not None
        retained = recovered.get(fresh["key_id"])
        assert retained is not None and retained["enabled"] is True
    assert JobJournal(path).replay().warnings == []


@pytest.mark.parametrize("torn", [False, True])
def test_corrupt_first_record_does_not_reenable_anonymous_mode(tmp_path: Path, torn: bool) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    old_secret, _ = store.mint()
    _damage_line(path, 0, torn=torn)
    for _ in range(3):
        store = _store(path)
        assert store.has_keys is True
        assert store.list() == []
        assert store.authenticate(old_secret) is None
    payloads = JobJournal(path).replay().payloads
    assert any(payload.get("auth_required") is True for payload in payloads)
    fresh_secret, _ = store.mint(admin=True)
    recovered = _store(path)
    assert recovered.authenticate(fresh_secret, required_scope="admin") is not None
    assert recovered.authenticate(old_secret) is None


def test_corrupt_rotation_never_restores_either_replaced_credential(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    old_secret, old = store.mint()
    new_secret, _ = store.rotate(old["key_id"])
    _damage_line(path, 1, torn=True)
    recovered = _store(path)
    assert recovered.authenticate(old_secret) is None
    assert recovered.authenticate(new_secret) is None
    assert recovered.has_keys is True


def test_request_quota_and_last_use_survive_restarts(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, key = store.mint(max_requests=2)
    assert store.authenticate(raw) is not None
    recovered = _store(path)
    snapshot = recovered.get(key["key_id"])
    assert snapshot is not None
    assert snapshot["uses"] == 1
    assert snapshot["last_used_at"] == 100.0
    assert recovered.authenticate(raw) is not None
    for _ in range(2):
        recovered = _store(path)
        with pytest.raises(KeyStoreError) as exhausted:
            recovered.authenticate(raw)
        assert exhausted.value.code == "quota_exceeded"


def test_token_quota_survives_restarts_and_revocation(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, key = store.mint(max_tokens=3)
    assert store.authenticate(raw) is not None
    store.charge_tokens(key["key_id"], 3)
    recovered = _store(path)
    with pytest.raises(KeyStoreError) as exhausted:
        recovered.authenticate(raw)
    assert exhausted.value.code == "quota_exceeded"
    recovered.revoke(key["key_id"])
    recovered.charge_tokens(key["key_id"], 2)
    recovered = _store(path)
    assert recovered.authenticate(raw) is None
    record = recovered.get(key["key_id"])
    assert record is not None
    assert record["tokens_used"] == 5
    assert record["uses"] == 1


def test_durable_snapshots_do_not_restore_rate_windows(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, _ = store.mint(rpm=1, max_requests=3)
    assert store.authenticate(raw, required_scope="read") is not None
    assert "_window_start" not in path.read_text()
    assert "_window_count" not in path.read_text()
    recovered = _store(path)
    assert recovered.authenticate(raw, required_scope="read") is not None
    with pytest.raises(KeyStoreError) as limited:
        recovered.authenticate(raw, required_scope="read")
    assert limited.value.code == "rate_limited"


def test_auth_snapshot_failure_does_not_authorize_or_consume_a_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, key = store.mint(rpm=1, max_requests=1)
    before = path.read_bytes()
    append = journal.append

    def fail(_payload: dict[str, Any]) -> None:
        raise OSError("injected snapshot failure")

    monkeypatch.setattr(journal, "append", fail)
    with pytest.raises(OSError, match="snapshot failure"):
        store.authenticate(raw, required_scope="read")
    assert path.read_bytes() == before
    assert store.get(key["key_id"]) == key
    monkeypatch.setattr(journal, "append", append)
    assert store.authenticate(raw, required_scope="read") is not None


def test_provider_charge_remains_live_when_snapshot_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, key = store.mint(max_tokens=1)

    def fail(_payload: dict[str, Any]) -> None:
        raise OSError("injected snapshot failure")

    monkeypatch.setattr(journal, "append", fail)
    with pytest.raises(OSError, match="snapshot failure"):
        store.charge_tokens(key["key_id"], 1)
    with pytest.raises(KeyStoreError) as exhausted:
        store.authenticate(raw)
    assert exhausted.value.code == "quota_exceeded"


def test_failed_repair_never_returns_an_authenticating_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    raw, key = store.mint()
    store.revoke(key["key_id"])
    _damage_line(path, 1, torn=True)
    damaged = path.read_bytes()
    journal = JobJournal(path)

    def fail(_payloads: list[dict[str, Any]]) -> None:
        raise OSError("injected repair failure")

    monkeypatch.setattr(journal, "compact", fail)
    with pytest.raises(OSError, match="repair failure"):
        ApiKeyStore(journal=journal, clock=lambda: 100.0)
    assert path.read_bytes() == damaged
    assert _store(path).authenticate(raw) is None


def test_untouched_store_still_creates_no_journal_file(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = _store(path)
    assert store.has_keys is False
    assert not path.exists()
