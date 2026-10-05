"""Rotation commits a whole replacement and policy updates consume iterables once."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fx1.serve import keys
from fx1.serve.journal import JobJournal
from fx1.serve.keys import KEY_PREFIX, ApiKeyStore


def test_revocation_write_failure_never_leaves_an_orphan_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, original = store.mint(name="previous", scopes=["read"], rpm=17)
    before = path.read_bytes()
    append = journal.append

    def fail_revocation(payload: dict[str, Any]) -> None:
        if payload.get("record", {}).get("enabled") is False:
            raise OSError("injected revocation append failure")
        append(payload)

    monkeypatch.setattr(journal, "append", fail_revocation)
    with pytest.raises(OSError, match="injected"):
        store.rotate(original["key_id"])

    assert len(store.list()) == 1
    assert path.read_bytes() == before
    assert store.authenticate(raw) is not None
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    assert len(recovered.list()) == 1
    assert recovered.authenticate(raw) is not None


def test_rotation_uses_one_verified_transition_and_replays_policy(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    raw, previous = store.mint(
        name="previous",
        scopes=["read"],
        rpm=17,
        ttl_s=60.0,
        max_requests=10,
        max_tokens=20,
    )
    successor, record = store.rotate(previous["key_id"])
    replay = JobJournal(path).replay()
    assert replay.warnings == []
    assert len(replay.payloads) == 2  # mint, then one whole rotation
    # An older single-record replayer must still honor the tombstone.
    assert replay.payloads[-1]["record"]["key_id"] == previous["key_id"]
    assert replay.payloads[-1]["record"]["enabled"] is False
    assert store.authenticate(raw) is None
    assert store.authenticate(successor) is not None
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    assert recovered.authenticate(raw) is None
    restored = recovered.authenticate(successor)
    assert restored is not None
    assert restored["rotated_from"] == previous["key_id"]
    for field in ("name", "scopes", "rpm", "max_requests", "max_tokens", "expires_at"):
        assert restored[field] == previous[field]
    assert record["rotated_from"] == previous["key_id"]


def test_process_loss_after_write_recovers_a_complete_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, previous = store.mint()
    append = journal.append

    class ProcessLost(BaseException):
        pass

    def append_then_stop(payload: dict[str, Any]) -> None:
        append(payload)
        raise ProcessLost

    monkeypatch.setattr(journal, "append", append_then_stop)
    monkeypatch.setattr(keys.secrets, "token_hex", lambda _n: "a" * 40)
    with pytest.raises(ProcessLost):
        store.rotate(previous["key_id"])

    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    assert recovered.authenticate(raw) is None
    successor = recovered.authenticate(KEY_PREFIX + "a" * 40)
    assert successor is not None
    assert successor["rotated_from"] == previous["key_id"]


def test_torn_rotation_never_recovers_two_enabled_secrets(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    raw, previous = store.mint()
    prefix = path.read_bytes()
    successor, _ = store.rotate(previous["key_id"])
    suffix = path.read_bytes()[len(prefix) :]
    cuts = {1, len(suffix) // 2, len(suffix) - 2, len(suffix)}
    offset = 0
    for line in suffix.splitlines(keepends=True):
        offset += len(line)
        cuts.add(offset)
    for cut in sorted(cuts):
        cut_path = tmp_path / f"cut_{cut}.jsonl"
        cut_path.write_bytes(prefix + suffix[:cut])
        recovered = ApiKeyStore(journal=JobJournal(cut_path), clock=lambda: 100.0)
        old_live = recovered.authenticate(raw) is not None
        new_live = recovered.authenticate(successor) is not None
        assert old_live != new_live, f"partial swap at rotation byte {cut}"


def test_keep_old_rotation_still_preserves_both_credentials(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    raw, previous = store.mint()
    successor, record = store.rotate(previous["key_id"], revoke_old=False)
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    assert recovered.authenticate(raw) is not None
    assert recovered.authenticate(successor) is not None
    assert record["rotated_from"] == previous["key_id"]


def test_clear_consumes_a_single_use_iterable_once(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    _, previous = store.mint(name="previous", rpm=17, max_requests=10)
    updated = store.update(previous["key_id"], clear=(field for field in ["name", "rpm"]))
    assert updated["name"] is None
    assert updated["rpm"] is None
    assert updated["max_requests"] == 10
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    restored = recovered.get(previous["key_id"])
    assert restored is not None
    assert restored["name"] is None
    assert restored["rpm"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_update_refuses_nonfinite_expiry_without_a_write(tmp_path: Path, value: float) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    _, previous = store.mint(ttl_s=60.0)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="expires_at"):
        store.update(previous["key_id"], expires_at=value)
    assert path.read_bytes() == before
    assert store.get(previous["key_id"]) == previous


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rotate_refuses_nonfinite_ttl_without_a_write(tmp_path: Path, value: float) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    _, previous = store.mint(ttl_s=60.0)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="ttl_s"):
        store.rotate(previous["key_id"], ttl_s=value)
    assert path.read_bytes() == before
    assert store.get(previous["key_id"]) == previous


def test_rotate_refuses_overflowed_fresh_expiry(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    now = [100.0]
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: now[0])
    _, previous = store.mint()
    before = path.read_bytes()
    now[0] = 1e308
    with pytest.raises(ValueError, match="expires_at"):
        store.rotate(previous["key_id"], ttl_s=1e308)
    assert path.read_bytes() == before
