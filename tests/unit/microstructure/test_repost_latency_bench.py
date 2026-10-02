"""Tests for repost_latency_bench."""

from quant_fund.microstructure.repost_latency_bench import repost_latency_bench


def test_repost_latency_shape() -> None:
    out = repost_latency_bench(horizon=400, seed=7)
    assert out["schema"] == "repost_latency.v1"
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
        "reseed_card_closed",
        "pins_survive",
        "kernel_carried",
    }
    assert len(out["receipt_sha256"]) == 64
