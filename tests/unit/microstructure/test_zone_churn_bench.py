"""Tests for zone_churn_bench + the maker_requote knob."""

from quant_fund.microstructure.reseed_hazard_bench import _calibrated
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.microstructure.zone_churn_bench import zone_churn_bench


def test_requote_inert_without_ttl() -> None:
    a = ZILobSimulator(_calibrated(7, {}))
    b = ZILobSimulator(_calibrated(7, {"maker_requote": 0.9}))
    for _ in range(400):
        a.step()
        b.step()
    assert a.n_fills == b.n_fills
    assert a.n_events == b.n_events
    assert a.best_bid_level == b.best_bid_level


def test_requote_churns() -> None:
    dead = ZILobSimulator(_calibrated(7, {"maker_ttl": 30}))
    churn = ZILobSimulator(_calibrated(7, {"maker_ttl": 30, "maker_requote": 0.9}))
    for _ in range(800):
        dead.step()
        churn.step()
    # Churned book keeps far more depth than pure aging.
    assert churn.total_depth > dead.total_depth
    assert churn.n_cancellations > dead.n_cancellations


def test_zone_churn_shape() -> None:
    out = zone_churn_bench(horizon=600, seed=7)
    assert out["schema"] == "zone_churn.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 8
    for c in out["cells"]:
        assert c["n_draws"] == 2
    assert set(out["claims"]) == {
        "cells_measured",
        "churn_closes_joint",
        "churn_lifts_pins",
        "churn_keeps_life",
        "kernel_carried",
    }
    assert len(out["receipt_sha256"]) == 64
