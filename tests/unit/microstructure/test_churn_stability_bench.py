"""Tests for churn_stability_bench."""

from quant_fund.microstructure.churn_stability_bench import churn_stability_bench


def test_churn_stability_shape() -> None:
    out = churn_stability_bench(horizon=400, seed=7)
    assert out["schema"] == "churn_stability.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 8  # 4 configs x 2 flows
    for c in out["cells"]:
        assert c["n_draws"] == 4
        assert 0.0 <= c["joint_closure_rate"] <= 1.0
        assert 0.0 <= c["all_seven_rate"] <= 1.0
        for rate in c["pin_rates"].values():
            assert 0.0 <= rate <= 1.0
    assert set(out["claims"]) == {
        "cells_measured",
        "closure_near_structural",
        "closure_multi_flow",
        "churn_is_mechanism",
    }
    assert len(out["receipt_sha256"]) == 64
