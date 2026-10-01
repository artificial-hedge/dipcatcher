"""Tests for closure_fit: score surface over the level-memory knobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.microstructure.closure_fit import (
    CLOSURE_FIT_SCHEMA,
    _measure_cell,
    _score,
    closure_fit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_GRID = [(0, 0.3, 3.0), (100, 0.5, 5.0)]
_TARGETS = {"instant": 0.887, "k200": 4.6447, "g1": 2.49, "spread": 13.09}


def _write_pair(
    tmp_path: Path,
    messages: list[list[Any]],
    book_rows: list[list[int]],
) -> None:
    msg = tmp_path / "AMZN_x_message_y.csv"
    ob = tmp_path / "AMZN_x_orderbook_y.csv"
    with msg.open("w", newline="") as f:
        csv.writer(f).writerows(messages)
    with ob.open("w", newline="") as f:
        csv.writer(f).writerows(book_rows)


def test_measure_cell_runs() -> None:
    m = _measure_cell(50, 0.3, 3.0, 2000, 5)
    assert m["n_fills"] > 0
    assert m["instant_signed_ticks"] is not None
    assert m["n_lo_suppressed"] >= 0


def test_score_normalized() -> None:
    metrics = {"instant": 0.887, "k200": 4.6447, "g1": 2.49, "spread": 13.09}
    assert _score(metrics, _TARGETS) == pytest.approx(0.0)
    metrics["instant"] = None
    assert _score(metrics, _TARGETS) is None


def test_bench_mini_grid_seals_and_verifies(tmp_path: Path) -> None:
    """Synthetic tape + 2-cell mini grid: shape, claims, sealed receipt."""
    _write_pair(
        tmp_path,
        [[34200.0, 1, 11, 100, 400000, 1]],
        [[4000, 100, 3900, 200]],
    )
    payload = closure_fit_bench(tmp_path, "AMZN", horizon=400, seed=3, grid=_GRID)
    assert payload["schema"] == CLOSURE_FIT_SCHEMA
    assert payload["kind"] == "closure_fit_bench"
    assert payload["data_label"] == "MIXED"
    assert payload["research_only"] is True
    assert len(payload["cells"]) == 2
    assert set(payload["targets"]) == {"instant", "k200", "g1", "spread"}
    claims = payload["claims"]
    assert "frontier_three_of_four" in claims
    assert "instant_continuation_antagonism" in claims
    assert "mechanism_gap" in claims
    assert payload["cells"][0]["score"] is None or isinstance(payload["cells"][0]["score"], float)
    body = dict(payload)
    sha = body.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(body)) == sha
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(payload))
    assert verify_receipt_file(receipt_path)["valid"] is True


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        closure_fit_bench(tmp_path, "AMZN", horizon=200, grid=_GRID)
