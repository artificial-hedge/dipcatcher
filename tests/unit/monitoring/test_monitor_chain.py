"""Tests for monitoring/monitor_chain.py — tamper-evident ops ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.monitoring.monitor_chain import MonitorChain


def _chain(tmp: Path) -> MonitorChain:
    return MonitorChain(tmp / "ops.jsonl")


def test_append_verify_roundtrip(tmp_path):
    c = _chain(tmp_path)
    c.append("drift", {"psi": 0.31, "feature": "spread"}, ts=1.0)
    c.append("kill_switch", {"from": "ENABLED", "to": "HALTED"}, ts=2.0)
    out = c.verify()
    assert out == {"ok": True, "n": 2, "errors": [], "tip": c.tip()}


def test_empty_chain_verifies(tmp_path):
    c = _chain(tmp_path)
    assert c.verify()["ok"]
    assert c.verify()["n"] == 0


def test_content_edit_detected(tmp_path):
    c = _chain(tmp_path)
    c.append("drift", {"psi": 0.31}, ts=1.0)
    c.append("gate", {"name": "x"}, ts=2.0)
    lines = (tmp_path / "ops.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["detail"]["psi"] = 0.01  # rewrite history
    lines[0] = json.dumps(rec, separators=(",", ":"))
    (tmp_path / "ops.jsonl").write_text("\n".join(lines) + "\n")
    out = c.verify()
    assert not out["ok"]
    assert any("content_edit@0" in e for e in out["errors"])


def test_deletion_breaks_chain(tmp_path):
    c = _chain(tmp_path)
    for i in range(3):
        c.append("other", {"i": i}, ts=float(i))
    lines = (tmp_path / "ops.jsonl").read_text().splitlines()
    (tmp_path / "ops.jsonl").write_text("\n".join([lines[0]] + lines[2:]) + "\n")
    out = c.verify()
    assert not out["ok"]
    assert any("chain_break" in e or "seq_gap" in e for e in out["errors"])


def test_malformed_line_fails_closed(tmp_path):
    (tmp_path / "ops.jsonl").write_text('{"seq":0\n')
    assert not _chain(tmp_path).verify()["ok"]


def test_bad_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        _chain(tmp_path).append("pnl", {"v": 1})


def test_tail(tmp_path):
    c = _chain(tmp_path)
    for i in range(5):
        c.append("dashboard", {"i": i})
    tail = c.tail(2)
    assert [r["detail"]["i"] for r in tail] == [3, 4]
