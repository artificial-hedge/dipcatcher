"""Tests for zone_embargo_bench."""

from quant_fund.microstructure.zone_embargo_bench import zone_embargo_bench


def test_zone_embargo_shape() -> None:
    out = zone_embargo_bench(horizon=1500, seed=7)
    assert out["schema"] == "zone_embargo.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 6
    for c in out["cells"]:
        assert c["n_draws"] == 2
        assert 0.0 <= c["share_9_63_mean"] <= 1.0
        for d in c["draws"]:
            assert 0 <= d["n_pins_ok"] <= 7
    assert set(out["claims"]) == {
        "cells_measured",
        "zone_reaches_occupancy",
        "zone_clears_tight",
        "zone_seven_pin_draw",
    }
    assert len(out["receipt_sha256"]) == 64


def test_zone_embargo_default_is_zero() -> None:
    from quant_fund.microstructure.zi_lob_simulator import ZILobConfig

    assert ZILobConfig(seed=1).zone_embargo == 0
    import pytest

    with pytest.raises(ValueError, match="zone_embargo"):
        ZILobConfig(seed=1, zone_embargo=-1)
