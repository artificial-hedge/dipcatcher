"""Tests for floor_stability_bench."""

from quant_fund.microstructure.floor_stability_bench import floor_stability_bench


def test_floor_stability_shape() -> None:
    out = floor_stability_bench(horizon=3000, seed=7)
    assert out["schema"] == "floor_stability.v1"
    assert out["research_only"] is True
    assert len(out["draws"]) == 8
    assert set(out["hold_rates"]) == {"g8_d320", "g8_d280_rp60"}
    for rates in out["hold_rates"].values():
        assert set(rates) == {
            "crown",
            "empty",
            "spread",
            "hidden",
            "reseed_rate",
            "reseed_touch",
            "reveal_gap",
            "all_seven",
        }
        assert all(0.0 <= r <= 1.0 for r in rates.values())
    assert set(out["claims"]) == {
        "cells_measured",
        "spread_stable",
        "reseed_stable",
        "stable_closure",
    }
    assert len(out["receipt_sha256"]) == 64
