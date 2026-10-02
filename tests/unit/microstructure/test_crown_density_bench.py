"""Tests for crown_density_bench — near-touch crown mass, tape vs sim."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.crown_density_bench import (
    CROWN_DENSITY_SCHEMA,
    _empty_gaps,
    _sim_crown,
    crown_density_bench,
)


def test_empty_gaps_masks_sentinels() -> None:
    import numpy as np

    bb = np.asarray([10, 10, -(10**9), 10])
    ba = np.asarray([12, 13, 13, 14])
    fills = np.asarray([1, 3])
    signs = np.asarray([1, -1])
    gaps = _empty_gaps(bb, ba, fills, signs, tick_units=1.0)
    # Buy fill at j=1 empties the ask: 12 -> 13, gap 1.0. Sell fill at
    # j=3 has a sentinel pre-fill bid (empty side) and is skipped.
    assert gaps == [1.0]


def test_sim_crown_fields() -> None:
    cell = _sim_crown("thin", None, horizon=3000, seed=3)
    assert cell["regime"] == "thin"
    for k in (
        "n_fills",
        "crown_units_mean",
        "visible_units_mean",
        "crown_share_of_visible",
        "reveal_gap_ticks_mean",
        "n_reveals",
        "spread_mean",
    ):
        assert k in cell


@pytest.mark.slow
def test_bench_sim_only() -> None:
    out = crown_density_bench(None, horizon=3000, seed=5)
    assert out["schema"] == CROWN_DENSITY_SCHEMA
    assert out["data_label"] == "SYNTHETIC"
    assert out["research_only"] is True
    assert len(out["cells"]) == 4
    assert out["receipt_sha256"]
