"""Tests for fx1.serve.journal_audit + the durable job journal."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fx1.serve.journal import JobJournal
from fx1.serve.journal_audit import journal_audit, journal_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = journal_audit()
    assert len(results) >= 10
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = journal_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    assert journal_audit_bench()["receipt_sha256"] == journal_audit_bench()["receipt_sha256"]


def test_replay_missing_file_is_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        res = JobJournal(Path(td) / "none.jsonl").replay()
        assert res.payloads == [] and res.truncated_at is None


def test_append_fsyncs_to_disk() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "j.jsonl"
        JobJournal(path).append({"a": 1})
        blob = path.read_bytes()
        assert b'"seq":0' in blob and b'"sha256":"' in blob
