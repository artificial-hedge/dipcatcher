"""Tests for zone_stability_bench."""

from quant_fund.microstructure.zone_stability_bench import zone_stability_bench


def test_zone_stability_shape() -> None:
    out = zone_stability_bench(horizon=1200, seed=7)
    assert out["schema"] == "zone_stability.v1"
    assert out["research_only"] is True
    assert len(out["panels"]) == 4
    assert len(out["impact_kernels"]) == 2
    for p in out["panels"]:
        assert p["n_draws"] == 4
        assert 0.0 <= p["all_seven_rate"] <= 1.0
        assert set(p["per_pin_rate"]) == {
            "crown",
            "empty",
            "spread",
            "hidden",
            "reseed_rate",
            "reseed_touch",
            "reveal_gap",
        }
    for k in out["impact_kernels"]:
        assert "instant_signed_ticks" in k
        assert "kernel_mean_ticks" in k
    assert set(out["claims"]) == {
        "cells_measured",
        "zone_closure_structural",
        "zone_closure_majority",
        "zone_carries_instant",
    }
    assert len(out["receipt_sha256"]) == 64
