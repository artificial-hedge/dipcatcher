"""Tests for churn_reseed_bench."""

from quant_fund.microstructure.churn_reseed_bench import churn_reseed_bench


def test_churn_reseed_shape() -> None:
    out = churn_reseed_bench(horizon=400, seed=7)
    assert out["schema"] == "churn_reseed.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 8  # 4 configs x 2 flows
    for c in out["cells"]:
        assert len(c["draws"]) == 2
        for d in c["draws"]:
            assert d["repost_fate_sums"]
            assert d["repost_due"] == d["repost_rested"] + sum(d["repost_drops"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "repost_fates_sum",
        "joint_reseed_in_band",
        "walked_past_rare",
        "repost_underfill",
    }
    assert len(out["receipt_sha256"]) == 64
