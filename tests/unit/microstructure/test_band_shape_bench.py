"""Tests for band_shape_bench."""

from quant_fund.microstructure.band_shape_bench import band_shape_bench


def test_band_shape_shape() -> None:
    out = band_shape_bench(horizon=3000, seed=7)
    assert out["schema"] == "band_shape.v1"
    assert out["research_only"] is True
    assert len(out["cells"]) == 5
    assert [c["density_exponent"] for c in out["cells"]] == [1.0, 2.0, 3.0, 2.0, 3.0]
    assert [c["flow_intensity"] for c in out["cells"]] == [None, None, None, 2.0, 2.0]
    for c in out["cells"]:
        assert c["n_pins_ok"] == sum(c["pins"].values())
    assert set(out["claims"]) == {
        "cells_measured",
        "shape_opens_spread",
        "spread_monotone_in_beta",
        "emptied_touch_survives",
    }
    assert len(out["receipt_sha256"]) == 64
