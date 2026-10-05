"""Expiry policy refuses nonfinite inputs and malformed journal recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_invalid_ttl_does_not_mint(tmp_path: Path, ttl: float) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)

    with pytest.raises(ValueError, match="ttl_s"):
        store.mint(ttl_s=ttl)

    assert store.list() == []
    assert not path.exists()


def test_finite_ttl_cannot_overflow_expiry(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 1e308)

    with pytest.raises(ValueError, match="expires_at"):
        store.mint(ttl_s=1e308)

    assert store.list() == []
    assert not path.exists()


@pytest.mark.parametrize(
    "expires",
    [float("nan"), float("inf"), float("-inf"), "never", [], {}, True, 10**1000],
    ids=["nan", "infinity", "negative_infinity", "string", "list", "dict", "bool", "huge_int"],
)
def test_malformed_recovered_expiry_fails_closed(tmp_path: Path, expires: Any) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 0.0)
    raw, record = store.mint(ttl_s=10.0)
    # This is a correctly hash-chained legacy record, not journal corruption.
    journal.append(
        {
            "record": {
                **record,
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "expires_at": expires,
            }
        }
    )
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 0.0)

    assert recovered.authenticate(raw) is None
    recovered_record = recovered.get(record["key_id"])
    assert recovered_record is not None
    assert recovered_record["uses"] == 0
    assert recovered_record["last_used_at"] is None


def test_finite_expiry_survives_restart_and_expires_at_boundary(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    raw, record = store.mint(ttl_s=10.0)
    now = [109.999]
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: now[0])

    assert recovered.authenticate(raw) is not None
    now[0] = 110.0
    assert recovered.authenticate(raw) is None
    recovered_record = recovered.get(record["key_id"])
    assert recovered_record is not None
    assert recovered_record["uses"] == 1
    assert recovered_record["expires_at"] == 110.0


def test_unbounded_legacy_record_remains_valid(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    journal = JobJournal(path)
    store = ApiKeyStore(journal=journal, clock=lambda: 100.0)
    raw, record = store.mint()
    record.pop("expires_at")
    journal.append({"record": {**record, "sha256": hashlib.sha256(raw.encode()).hexdigest()}})
    recovered = ApiKeyStore(journal=JobJournal(path), clock=lambda: 1e20)

    assert recovered.authenticate(raw) is not None
