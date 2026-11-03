"""The chain head is appended to an unsigned anchors file and prod-backup copies it."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund.api.admin_audit import admin_context, audit_log_env_names, note_mutation
from quant_fund.api.http_controls import key_file_env_names

_ROOT = Path(__file__).resolve().parents[3]
_ANCHOR = _ROOT / "scripts" / "anchor_audit_head.py"
_BACKUP = _ROOT / "scripts" / "prod_backup.py"
_ADMIN = "anchor-admin-key"
_GENESIS = hashlib.sha256(b"").hexdigest()


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in key_file_env_names():
        monkeypatch.delenv(name, raising=False)
    for name in audit_log_env_names():
        monkeypatch.delenv(name, raising=False)


def _run(script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode("ascii")).hexdigest()


def test_anchor_matches_the_next_prev_and_backup_copies_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    anchors = tmp_path / "anchors.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    monkeypatch.setenv("QUANT_API_AUDIT_LOG_MAX_BYTES", "500")
    with admin_context("cli:anchor", _ADMIN):
        note_mutation("config", "written")
    first = _run(_ANCHOR, [str(log_path), "--anchors", str(anchors)])
    assert first.returncode == 0
    assert first.stdout.strip() == "audit anchor ok"
    assert _ADMIN not in first.stdout
    assert _ADMIN not in first.stderr
    opened = json.loads(anchors.read_text(encoding="utf-8"))
    last = log_path.read_text(encoding="utf-8").splitlines()[-1]
    assert opened["head"] == _digest(last)
    assert opened["time"].endswith("Z")
    assert set(opened) == {"head", "time"}
    assert (anchors.stat().st_mode & 0o777) == 0o600
    assert _ADMIN not in anchors.read_text(encoding="utf-8")
    with admin_context("cli:anchor-2", _ADMIN):
        note_mutation("ledger", "written")
    chained = log_path.read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(chained)["prev"] == opened["head"]
    for _step in range(6):
        with admin_context("cli:rotate", _ADMIN):
            note_mutation("config", "written")
    older = log_path.with_name(log_path.name + ".1")
    assert older.is_file()
    second = _run(_ANCHOR, [str(log_path), "--anchors", str(anchors)])
    assert second.returncode == 0
    rows = [json.loads(line) for line in anchors.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    head_line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    assert rows[1]["head"] == _digest(head_line)
    assert _ADMIN not in anchors.read_text(encoding="utf-8")
    dest = tmp_path / "backup"
    copied = _run(
        _BACKUP,
        ["--audit-log", str(log_path), "--anchors", str(anchors), "--dest", str(dest)],
    )
    assert copied.returncode == 0
    assert copied.stdout.splitlines()[0] == "prod backup ok"
    assert _ADMIN not in copied.stdout
    names = set(copied.stdout.splitlines()[1:])
    present = {
        path.name
        for path in tmp_path.iterdir()
        if path.name == anchors.name
        or path.name == log_path.name
        or path.name.startswith(f"{log_path.name}.")
    }
    assert names == present
    assert older.name in names
    for name in names:
        assert (dest / name).read_bytes() == (tmp_path / name).read_bytes()
        assert ((dest / name).stat().st_mode & 0o777) == 0o600
    absent = _run(_ANCHOR, [str(tmp_path / "missing.jsonl"), "--anchors", str(anchors)])
    assert absent.returncode == 1
    assert absent.stdout.strip() == "audit anchor missing"
    blocked = tmp_path / "anchors-dir"
    blocked.mkdir()
    unavailable = _run(_ANCHOR, [str(log_path), "--anchors", str(blocked)])
    assert unavailable.returncode == 1
    assert unavailable.stdout.strip() == "audit anchor unavailable"


def test_an_empty_chain_anchors_the_genesis_digest(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    anchors = tmp_path / "anchors.jsonl"
    log_path.write_text("", encoding="utf-8")
    result = _run(_ANCHOR, [str(log_path), "--anchors", str(anchors)])
    assert result.returncode == 0
    assert json.loads(anchors.read_text(encoding="utf-8"))["head"] == _GENESIS
    missing = _run(
        _BACKUP,
        [
            "--audit-log",
            str(log_path),
            "--anchors",
            str(tmp_path / "nope"),
            "--dest",
            str(tmp_path / "out"),
        ],
    )
    assert missing.returncode == 1
    assert missing.stdout.strip() == "prod backup missing"
