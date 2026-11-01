"""Schema-drift lane: key-set consistency + revision provenance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from quant_fund.registry import schema_drift as schema_drift_mod
from quant_fund.registry.schema_drift import schema_drift, schema_drift_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def _mk(d: Path, name: str, tag: str, keys: list[str], rev: str | None = None) -> None:
    payload = {"kind": tag, "claim": dict.fromkeys(keys)}
    if rev:
        payload["git_revision"] = rev
    (d / name).write_text(json.dumps(payload))


def test_drift_detected_within_tag(tmp_path: Path) -> None:
    _mk(tmp_path, "a.json", "t.v1", ["x", "y"])
    _mk(tmp_path, "b.json", "t.v1", ["x", "y"])
    _mk(tmp_path, "c.json", "t.v1", ["x"])  # dropped 'y'
    audit = schema_drift(tmp_path)
    assert audit["groups_with_drift"] == 1
    assert audit["drifted"][0]["file"] == "c.json"
    assert audit["drifted"][0]["missing"] == ["y"]


def test_phantom_revision_not_a_commit_flags(tmp_path: Path) -> None:
    _mk(tmp_path, "a.json", "t.v1", ["x"], rev="deadbeef")
    audit = schema_drift(tmp_path, repo=".")
    assert audit["n_flagged"] == 1
    assert audit["phantom_revisions"][0]["why"] == "not_a_commit"


def test_git_timeout_is_not_treated_as_an_ancestor(tmp_path: Path, monkeypatch) -> None:
    _mk(tmp_path, "a.json", "t.v1", ["x"], rev="a" * 40)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") == schema_drift_mod._GIT_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd=args, timeout=schema_drift_mod._GIT_TIMEOUT_S)

    monkeypatch.setattr(schema_drift_mod.subprocess, "run", fake_run)
    audit = schema_drift(tmp_path, repo=tmp_path)
    assert audit["n_flagged"] == 1
    assert audit["phantom_revisions"][0]["why"] == "not_a_commit"
    assert audit["phantom_revisions"][0]["file"] == "a.json"


def test_corpus_clean_and_bench_verifies() -> None:
    receipt = schema_drift_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "schema_drift_test.json")
    assert result["valid"], result.get("errors")
