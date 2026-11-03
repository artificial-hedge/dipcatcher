"""Runtime credential files are mode 0600 and fail closed when world-readable."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from fx1.interactive import profiles
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore

_MARKER = "marker-not-a-live-credential"


def test_keystore_write_and_reload_stay_private(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    store = ApiKeyStore(journal=JobJournal(path))
    store.mint("label")
    reloaded = ApiKeyStore(journal=JobJournal(path))
    assert reloaded.has_keys
    if os.name == "nt":
        return
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_world_readable_keystore_is_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "keys.jsonl"
    path.write_text(_MARKER + "\n", encoding="utf-8")
    if os.name == "nt":
        return
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="world-readable") as raised:
        ApiKeyStore(journal=JobJournal(path))
    assert _MARKER not in str(raised.value)


def test_ordinary_journal_is_not_treated_as_a_keystore(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    path.write_text("torn\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o644)
    result = JobJournal(path).replay()
    assert result.dropped == 1


def test_credential_store_refuses_a_world_readable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    path = profiles.set_endpoint("fx1", _MARKER)
    if os.name == "nt":
        return
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) & stat.S_IRWXO == 0
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="world-readable") as raised:
        profiles.resolve_endpoint("fx1")
    assert _MARKER not in str(raised.value)
