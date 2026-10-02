"""Tests for floor_pins_bench."""

from quant_fund.microstructure.floor_pins_bench import floor_pins_bench


def test_floor_pins_shape() -> None:
    out = floor_pins_bench(horizon=3000, seed=7)
    assert out["schema"] == "floor_pins.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 6
    assert [c["min_quote_dist"] for c in out["cells"]] == [12, 12, 12, 14, 14, 10]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "all_pins_at_corner",
        "composes_under_both_flows",
        "empty_survives_dose",
    }
    assert len(out["receipt_sha256"]) == 64
