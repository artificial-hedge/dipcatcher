"""Tests for iid_floor_bench."""

from quant_fund.microstructure.iid_floor_bench import iid_floor_bench


def test_iid_floor_shape() -> None:
    out = iid_floor_bench(horizon=2500, seed=7)
    assert out["schema"] == "iid_floor.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    for c in out["cells"]:
        assert c["n_draws"] == 3
        assert 0.0 <= c["spread_rate"] <= 1.0
        assert 0.0 <= c["all_seven_rate"] <= 1.0
        assert len(c["draws"]) == 3
        for d in c["draws"]:
            assert d["n_pins_ok"] == sum(d["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "iid_structural",
        "iid_seven_pin_draw",
        "iid_dose_response",
    }
    assert len(out["receipt_sha256"]) == 64
