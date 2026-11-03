"""The admin audit file is a hash chain, including after rotation."""

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
_SCRIPT = _ROOT / "scripts" / "verify_audit_chain.py"
_ADMIN = "chain-admin-key"


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in key_file_env_names():
        monkeypatch.delenv(name, raising=False)
    for name in audit_log_env_names():
        monkeypatch.delenv(name, raising=False)


def _verify(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_rotated_lines_stay_one_chain_and_a_tear_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    monkeypatch.setenv("QUANT_API_AUDIT_LOG_MAX_BYTES", "500")
    for step in range(8):
        with admin_context(f"cli:step-{step}", _ADMIN):
            note_mutation("config", "written")
    older = log_path.with_name(log_path.name + ".1")
    assert older.is_file()
    assert log_path.is_file()
    fresh = _verify(log_path)
    assert fresh.returncode == 0
    assert fresh.stdout.strip() == "audit chain ok"
    assert _ADMIN not in older.read_text(encoding="utf-8")
    assert _ADMIN not in log_path.read_text(encoding="utf-8")
    lines = older.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["outcome"] = "tampered"
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    older.write_text("\n".join(lines) + "\n", encoding="utf-8")
    torn = _verify(log_path)
    assert torn.returncode == 1
    assert torn.stdout.splitlines()[0] == "audit chain break"
    assert "prev mismatch" in torn.stdout
    missing = _verify(tmp_path / "absent.jsonl")
    assert missing.returncode == 1
    assert missing.stdout.splitlines()[0] == "audit chain break"


def test_a_leading_seed_line_is_the_first_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text("seed\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    with admin_context("cli:seed", _ADMIN):
        note_mutation("config", "written")
    checked = _verify(log_path)
    assert checked.returncode == 0
    assert checked.stdout.strip() == "audit chain ok"
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[1])
    assert row["prev"] == hashlib.sha256(b"seed").hexdigest()


def test_the_first_line_points_at_the_empty_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear(monkeypatch)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("QUANT_API_AUDIT_LOG", str(log_path))
    with admin_context("cli:first", _ADMIN):
        note_mutation("ledger", "written")
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["prev"] == hashlib.sha256(b"").hexdigest()
    assert row["kind"] == "ledger"
    assert _verify(log_path).returncode == 0
