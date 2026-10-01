"""Tests for microstructure/impact_law.py — metaorder impact exponent."""

from __future__ import annotations

import math

import pytest

from quant_fund.microstructure.impact_law import (
    IMPACT_LAW_SCHEMA,
    ImpactPoint,
    fit_power_law,
    impact_curve,
    impact_law_bench,
    probe_impact,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def _warm(beta: float = 0.0, seed: int = 0) -> ZILobSimulator:
    sim = ZILobSimulator(ZILobConfig(seed=seed, density_exponent=beta))
    for _ in range(300):
        sim.step()
    return sim


def test_probe_impact_buy_positive_and_walks() -> None:
    sim = _warm(0.0)
    imp, walk = probe_impact(sim, side="buy", qty=4)
    assert imp > 0
    assert walk >= 1


def test_probe_impact_sell_positive_normalized() -> None:
    # impact is direction-normalized: a sell walking the bid book down
    # reports positive magnitude
    sim = _warm(0.0, seed=2)
    imp, _ = probe_impact(sim, side="sell", qty=4)
    assert imp > 0


def test_fit_power_law_recovers_known_exponent() -> None:
    # I(Q) = 2.0 * Q^0.5 exactly
    pts = [
        ImpactPoint(
            qty=q,
            impact_ticks_mean=2.0 * math.sqrt(q),
            impact_ticks_std=0.0,
            levels_walked_mean=1.0,
            n_trials=1,
        )
        for q in (1, 4, 9, 16, 25)
    ]
    psi, c, r2 = fit_power_law(pts)
    assert psi == pytest.approx(0.5, abs=1e-9)
    assert c == pytest.approx(2.0, rel=1e-9)
    assert r2 == pytest.approx(1.0)


def test_fail_closed_edges() -> None:
    sim = _warm(0.0)
    with pytest.raises(ValueError):
        probe_impact(sim, side="buy", qty=0)
    with pytest.raises(ValueError):
        impact_curve(qty_grid=(), density_exponent=0.0, n_trials=2, warmup=10)
    with pytest.raises(ValueError):
        fit_power_law(
            [
                ImpactPoint(1, 1.0, 0.0, 1.0, 1),
                ImpactPoint(2, 1.2, 0.0, 1.0, 1),
            ]
        )
    with pytest.raises(ValueError):
        fit_power_law(
            [
                ImpactPoint(1, -1.0, 0.0, 1.0, 1),
                ImpactPoint(2, 1.2, 0.0, 1.0, 1),
                ImpactPoint(4, 1.5, 0.0, 1.0, 1),
            ]
        )


def test_impact_curve_monotone_in_size_ballpark() -> None:
    pts = impact_curve(qty_grid=(1, 4, 8), density_exponent=0.0, n_trials=4, warmup=200)
    assert len(pts) == 3
    assert pts[-1].impact_ticks_mean > pts[0].impact_ticks_mean


def test_bench_schema_determinism_and_scaling() -> None:
    r1 = impact_law_bench(qty_grid=(1, 2, 4, 8, 16), n_trials=4, warmup=200)
    r2 = impact_law_bench(qty_grid=(1, 2, 4, 8, 16), n_trials=4, warmup=200)
    assert r1["payload_sha256"] == r2["payload_sha256"]
    assert r1["schema"] == IMPACT_LAW_SCHEMA
    assert r1["kind"] == "impact_law"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    # Donier prediction holds on ordering in both fits
    assert r1["psi_ordering_holds"] is True
    if r1["subsat_ordering_holds"] is not None:
        assert r1["subsat_ordering_holds"] is True
    b0, b1 = r1["cells"]["beta_0"], r1["cells"]["beta_1"]
    assert 0.0 < b0["psi_hat"] < 1.2
    assert 0.0 < b1["psi_hat"] < b0["psi_hat"]
