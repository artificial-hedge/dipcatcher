"""Tests for joint_tune_bench."""

from quant_fund.microstructure.joint_tune_bench import joint_tune_bench


def test_joint_tune_shape() -> None:
    out = joint_tune_bench(horizon=400, seed=7)
    assert out["schema"] == "joint_tune.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 8
    for c in out["cells"]:
        assert c["n_draws"] == 2
        assert 0.0 <= c["n_pins_mean"] <= 7.0
        for rate in c["pin_rates"].values():
            assert 0.0 <= rate <= 1.0
    assert set(out["claims"]) == {
        "cells_measured",
        "joint_closure_found",
        "grammar_keeps_pins",
        "kernel_carried",
    }
    assert len(out["receipt_sha256"]) == 64
