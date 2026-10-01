"""Tests for anchor_scan_bench — the (gain × halflife) calibration surface."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.anchor_scan_bench import (
    ANCHOR_SCAN_SCHEMA,
    anchor_scan_bench,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


@pytest.fixture(scope="module")
def receipt() -> dict:
    return anchor_scan_bench(horizon=3000, seed=3)


def test_smoke_and_seal(receipt: dict) -> None:
    assert receipt["schema"] == ANCHOR_SCAN_SCHEMA
    assert receipt["kind"] == "anchor_scan_bench"
    assert receipt["research_only"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    seal = receipt.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(receipt)) == seal


def test_grid_shape(receipt: dict) -> None:
    cells = receipt["cells"]
    assert len(cells) == 2 * len(receipt["gains"]) * len(receipt["halflives"])
    seen = {(c["flow"], c["ref_fill_gain"], c["ref_halflife"]) for c in cells}
    assert len(seen) == len(cells)


def test_determinism() -> None:
    a = anchor_scan_bench(horizon=1200, seed=9)
    b = anchor_scan_bench(horizon=1200, seed=9)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_divergences_are_strings(receipt: dict) -> None:
    assert all(isinstance(d, str) for d in receipt["divergences"])


def test_best_cell_consistent(receipt: dict) -> None:
    cells = [c for c in receipt["cells"] if c["kernel_rmse_vs_real"] is not None]
    best = min(cells, key=lambda c: c["kernel_rmse_vs_real"])
    assert receipt["best_cell"]["flow"] == best["flow"]
    assert receipt["best_cell"]["ref_fill_gain"] == best["ref_fill_gain"]
