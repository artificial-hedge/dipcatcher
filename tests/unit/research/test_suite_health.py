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
