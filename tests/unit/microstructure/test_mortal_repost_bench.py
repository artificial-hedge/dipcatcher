"""Tests for mortal_repost_bench."""

from quant_fund.microstructure.mortal_repost_bench import mortal_repost_bench


def test_mortal_repost_shape() -> None:
    out = mortal_repost_bench(horizon=400, seed=7)
    assert out["schema"] == "mortal_repost.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 11
    for c in out["cells"]:
        assert c["n_draws"] == 2
        assert 0.0 <= c["n_pins_mean"] <= 7.0
    assert set(out["claims"]) == {
        "cells_measured",
        "mortal_damps_overshoot",
        "immune_damps_rate",
        "joint_closure_found",
        "pins_survive_mortal",
    }
    assert len(out["receipt_sha256"]) == 64
