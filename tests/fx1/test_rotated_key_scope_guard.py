"""Rotation preserves authorization before accounting, including after replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore, KeyStoreError


@pytest.mark.parametrize("restart", [False, True])
def test_rotated_key_scope_refusal_does_not_spend_its_budget(tmp_path: Path, restart: bool) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)
    _, original = store.mint(scopes=["read"], rpm=1, max_requests=1)
    secret, successor = store.rotate(original["key_id"])
    if restart:
        store = ApiKeyStore(journal=JobJournal(path), clock=lambda: 100.0)

    with pytest.raises(KeyStoreError) as denied:
        store.authenticate(secret, required_scope="write")
    assert denied.value.code == "insufficient_scope"
    record = store.get(successor["key_id"])
    assert record is not None
    assert record["uses"] == 0
    assert store.authenticate(secret, required_scope="read") is not None
    with pytest.raises(KeyStoreError) as exhausted:
        store.authenticate(secret, required_scope="read")
    assert exhausted.value.code == "quota_exceeded"
