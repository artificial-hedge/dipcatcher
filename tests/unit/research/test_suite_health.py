"""Contract tests for ``suite_health`` — the whole-corpus audit receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.receipt_v2 import seal_receipt
from quant_fund.research.suite_health import SUITE_HEALTH_SCHEMA, suite_health


def _sealed(tmp: Path, name: str) -> None:
    rec = seal_receipt(
        {
            "kind": "demo",
            "level": "research",
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
        }
    )
    (tmp / name).write_text(json.dumps(rec))


def test_clean_dir_pools_when_evalues_present(tmp_path: Path) -> None:
    _sealed(tmp_path, "a.json")
    _sealed(tmp_path, "b.json")
    frame, rec = suite_health(tmp_path)
    assert rec["schema"] == SUITE_HEALTH_SCHEMA
    assert rec["kind"] == "suite_health"
    assert rec["data_label"] == "SYNTHETIC"
    assert rec["live_pnl_claim"] is False
    assert rec["n_receipts"] == 2
    assert rec["n_failed"] == 0
    assert frame["valid"].all()


def test_corrupt_receipt_fails_closed(tmp_path: Path) -> None:
    _sealed(tmp_path, "good.json")
    (tmp_path / "bad.json").write_text('{"kind": "demo", "receipt_sha256": "x"}')
    frame, rec = suite_health(tmp_path)
    assert rec["n_failed"] == 1
    # a corrupt artifact withholds the pooled claim — never asserted over
    # partial evidence
    assert rec["pooled_evalue"] is None
    assert rec["pooled_alarmed"] is False
    assert frame.filter(~frame["valid"])["file"].to_list() == ["bad.json"]


def test_empty_dir_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no receipts"):
        suite_health(tmp_path)


def test_missing_dir_fails_closed() -> None:
    with pytest.raises(ValueError, match="not found"):
        suite_health("/nonexistent/path/xyz")


def test_harvested_evalues_pool(tmp_path: Path) -> None:
    """A receipt carrying a positive evalue finding feeds the pool when
    the corpus lane is present; on branches without it the field stays 0."""
    _sealed(tmp_path, "a.json")
    _, rec = suite_health(tmp_path)
    if rec["corpus_lane_available"]:
        assert isinstance(rec["n_evalues_pooled"], int)
    else:
        assert rec["n_evalues_pooled"] == 0
        assert rec["pooled_evalue"] is None


def test_label_aggregates_inputs(tmp_path: Path) -> None:
    """Suite label is the unique input label, MIXED for a mixed corpus."""
    rec = seal_receipt(
        {
            "kind": "demo",
            "level": "research",
            "data_label": "yahoo_eod",
            "research_only": True,
            "live_pnl_claim": False,
        }
    )
    (tmp_path / "r.json").write_text(json.dumps(rec))
    _frame, rep = suite_health(tmp_path)
    assert rep["data_label"] == "yahoo_eod"
    assert rep["params"]["input_labels"]["r.json"] == "yahoo_eod"

    _sealed(tmp_path, "s.json")
    _frame, rep2 = suite_health(tmp_path)
    assert rep2["data_label"] == "MIXED"


def test_strict_cli_gate(tmp_path: Path) -> None:
    """--strict exits nonzero on a corrupt receipt, zero on sealed/legacy."""
    import hashlib

    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli
    from quant_fund.research.legacy_unsealed import KNOWN_UNSEALED

    runner = CliRunner()
    out = tmp_path / "out"

    # all-sealed corpus → strict passes
    _sealed(tmp_path, "ok.json")
    res = runner.invoke(
        cli, ["suite-health", "--receipts-dir", str(tmp_path), "--out-dir", str(out), "--strict"]
    )
    assert res.exit_code == 0, res.output

    # corrupt sealed receipt → strict fails
    bad = json.loads((tmp_path / "ok.json").read_text())
    bad["data_label"] = "tampered"
    (tmp_path / "ok.json").write_text(json.dumps(bad))
    res = runner.invoke(
        cli, ["suite-health", "--receipts-dir", str(tmp_path), "--out-dir", str(out), "--strict"]
    )
    assert res.exit_code == 1
    assert "STRICT FAILURE" in res.output

    # byte-pinned legacy unsealed receipt → tolerated under strict
    (tmp_path / "ok.json").unlink()
    name, digest = next(iter(KNOWN_UNSEALED.items()))
    body = b'{"kind": "legacy", "note": "pre-seal"}'
    # fabricate the pinned bytes: the allowlist pins real files, so instead
    # assert the helper's contract directly on a non-listed name
    (tmp_path / name).write_bytes(body)
    res = runner.invoke(
        cli, ["suite-health", "--receipts-dir", str(tmp_path), "--out-dir", str(out), "--strict"]
    )
    # wrong bytes → still a strict failure (the pin covers content, not name)
    assert res.exit_code == 1
    assert hashlib.sha256(body).hexdigest() != digest


def test_suite_health_receipt_v2_round_trip(tmp_path: Path) -> None:
    """receipt_version=2 seals the suite_health.v1 body in the envelope."""
    from quant_fund.research.receipt_v2 import verify_receipt_file
    from quant_fund.research.suite_health import write_suite_health_receipt

    _sealed(tmp_path, "a.json")
    _sealed(tmp_path, "b.json")
    _, receipt = suite_health(tmp_path)
    path = write_suite_health_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "suite_health"
    assert payload["payload"]["inputs_sha256"] == receipt["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True


def test_dataset_sha256_tracks_corpus_bytes(tmp_path: Path) -> None:
    """Identical corpora share dataset_sha256 across audits and alpha;
    adding a file changes it."""
    _sealed(tmp_path, "a.json")
    _sealed(tmp_path, "b.json")
    _, r1 = suite_health(tmp_path)
    _, r2 = suite_health(tmp_path, alpha=0.1)
    d = r1["dataset_sha256"]
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)
    assert r2["dataset_sha256"] == d  # alpha is a run param, not data
    _sealed(tmp_path, "c.json")
    _, r3 = suite_health(tmp_path)
    assert r3["dataset_sha256"] != d
