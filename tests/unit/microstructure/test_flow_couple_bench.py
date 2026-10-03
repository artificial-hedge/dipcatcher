"""Tests for flow_couple_bench."""

from quant_fund.microstructure.flow_couple_bench import flow_couple_bench


def test_flow_couple_shape() -> None:
    out = flow_couple_bench(horizon=4000, seed=7)
    assert out["schema"] == "flow_couple.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 4
    assert [c["flow_intensity"] for c in out["cells"]] == [None, 1.0, 2.0, 3.0]
    for c in out["cells"]:
        assert set(c["pins"]) == {
            "crown",
            "empty",
            "hidden",
            "reseed_rate",
            "reseed_touch",
            "reveal_gap",
            "spread",
        }
        assert c["n_pins_ok"] == sum(c["pins"].values())
        assert set(c["kernel_mean_ticks"]) == {"1", "5", "20", "50", "200"}
        assert c["all_closed"] == bool(c["kernel_in_band"] and all(c["pins"].values()))
    assert set(out["claims"]) == {
        "cells_measured",
        "pins_hold_at_iid",
        "pins_survive_flow",
        "joint_closure_found",
        "kernel_scales_with_intensity",
    }
    assert len(out["receipt_sha256"]) == 64
