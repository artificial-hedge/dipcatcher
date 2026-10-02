"""Tests for floor_rate_bench."""

from quant_fund.microstructure.floor_rate_bench import floor_rate_bench


def test_floor_rate_shape() -> None:
    out = floor_rate_bench(horizon=3000, seed=7)
    assert out["schema"] == "floor_rate.v1"
    assert out["research_only"] is True
    assert len(out["draws"]) == 36  # 6 floors x 2 flows x 3 seeds
    assert len(out["cell_rates"]) == 12
    assert len(out["floor_rates"]) == 6
    for c in out["cell_rates"]:
        assert 0.0 <= c["band_rate"] <= 1.0
        assert c["n_draws"] == 3
    for d in out["draws"]:
        if d["in_band"]:
            assert 9.0 <= d["spread_mean"] <= 63.0
    assert set(out["claims"]) == {
        "cells_measured",
        "floor_structural",
        "opens_under_iid",
        "rate_reaches_half",
    }
    assert len(out["receipt_sha256"]) == 64
