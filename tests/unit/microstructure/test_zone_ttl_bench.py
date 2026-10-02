"""Tests for zone_ttl_bench + the maker_ttl knob."""

from quant_fund.microstructure.reseed_hazard_bench import _calibrated
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.microstructure.zone_ttl_bench import zone_ttl_bench


def test_ttl_zero_bit_identical() -> None:
    a = ZILobSimulator(_calibrated(7, {}))
    b = ZILobSimulator(_calibrated(7, {"maker_ttl": 0}))
    for _ in range(400):
        a.step()
        b.step()
    assert a.n_fills == b.n_fills
    assert a.best_bid_level == b.best_bid_level
    assert a.n_events == b.n_events


def test_ttl_validates() -> None:
    import pytest

    with pytest.raises(ValueError):
        _calibrated(7, {"maker_ttl": -1})


def test_ttl_fires() -> None:
    plain = ZILobSimulator(_calibrated(7, {}))
    aged = ZILobSimulator(_calibrated(7, {"maker_ttl": 30}))
    for _ in range(500):
        plain.step()
        aged.step()
    assert aged.n_cancellations > plain.n_cancellations


def test_zone_ttl_shape() -> None:
    out = zone_ttl_bench(horizon=800, seed=7)
    assert out["schema"] == "zone_ttl.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 10
    for c in out["cells"]:
        assert c["n_draws"] == 2
        for d in c["draws"]:
            assert 0 <= d["n_pins"] <= 7
    assert set(out["claims"]) == {
        "cells_measured",
        "ttl_reaches_life_scale",
        "ttl_improves_mix",
        "ttl_keeps_closure",
        "kernel_carried",
    }
    assert len(out["receipt_sha256"]) == 64
