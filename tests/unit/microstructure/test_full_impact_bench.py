"""Tests for full_impact_bench."""

from quant_fund.microstructure.full_impact_bench import full_impact_bench


def test_full_impact_shape() -> None:
    out = full_impact_bench(horizon=6000, seed=7)
    assert out["schema"] == "full_impact.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 5
    names = [c["regime"] for c in out["cells"]]
    assert names == ["deep", "joint", "joint_split", "full", "full_split"]
    for c in out["cells"]:
        assert set(c["kernel_mean_ticks"]) == {"1", "5", "20", "50", "200"}
    assert set(out["claims"]) == {
        "cells_measured",
        "full_stack_carries_instant",
        "full_stack_carries_continuation",
        "repost_preserves_kernel",
    }
    assert len(out["receipt_sha256"]) == 64
