"""Tests for flee_wide_bench — the wide-regime flee×chase composition."""

from __future__ import annotations

import pytest

from quant_fund.microstructure.flee_wide_bench import (
    _CELLS,
    FLEE_WIDE_SCHEMA,
    _cell,
    flee_wide_bench,
)


def test_cell_fields_and_regime() -> None:
    cell = _cell("wide", 0.10, 2, 50, 0.5, horizon=3000, seed=3)
    assert cell["regime"] == "wide"
    assert cell["hit_flee_frac"] == 0.10
    assert cell["n_fills"] >= 0
    for k in (
        "spread_mean",
        "spread_occupancy_9_21",
        "instant_signed_ticks",
        "instant_given_empty_ticks",
        "instant_given_kept_ticks",
        "k200_ticks",
        "lo_channel_ticks",
        "fill_channel_ticks",
        "cxl_channel_ticks",
        "n_hit_flees",
    ):
        assert k in cell


def test_wide_book_is_wide() -> None:
    ref = _cell("wide", 0.0, 0, 0, 0.5, horizon=3000, seed=5)
    assert ref["spread_mean"] is not None and ref["spread_mean"] > 5.0


@pytest.mark.slow
def test_bench_smoke() -> None:
    out = flee_wide_bench(horizon=3000, seed=5)
    assert out["schema"] == FLEE_WIDE_SCHEMA
    assert out["kind"] == "sim_bench"
    assert out["data_label"] == "SYNTHETIC"
    assert out["research_only"] is True
    assert len(out["cells"]) == len(_CELLS)
    assert set(out["claims"]) == {
        "grid_evaluated",
        "wide_regime_reached",
        "flee_cxl_closer_wide",
        "wide_over_reveals",
    }
    assert out["receipt_sha256"]
