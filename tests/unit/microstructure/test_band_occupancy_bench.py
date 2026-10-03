"""Tests for band_occupancy_bench."""

from quant_fund.microstructure.band_occupancy_bench import band_occupancy_bench


def test_band_occupancy_shape() -> None:
    out = band_occupancy_bench(horizon=2500, seed=7)
    assert out["schema"] == "band_occupancy.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 5
    assert out["tape_reference"]["share_9_63"] == 0.8039
    for c in out["cells"]:
        assert c.get("n_obs", 0) > 0
        assert 0.0 <= c["share_9_63"] <= 1.0
        assert 0.0 <= c["tight_share_le2"] <= 1.0
    assert set(out["claims"]) == {
        "cells_measured",
        "floor_reaches_occupancy",
        "tight_regime_cleared",
        "median_on_tape",
    }
    assert len(out["receipt_sha256"]) == 64
