"""Tests for microstructure/pin_estimator.py — Easley-O'Hara PIN MLE."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.pin_estimator import (
    PIN_SCHEMA,
    fit_pin,
    log_likelihood_day,
    pin_estimator_bench,
    pin_of,
    simulate_pin_days,
)


def test_log_likelihood_day_matches_brute_force() -> None:
    """Lin-Ke factorization must equal the naive mixture at small counts."""
    a, d, m, e = 0.4, 0.3, 20.0, 15.0
    b, s = 18.0, 12.0

    def pois(k: float, lam: float) -> float:
        return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

    naive = (
        (1 - a) * pois(b, e) * pois(s, e)
        + a * (1 - d) * pois(b, e + m) * pois(s, e)
        + a * d * pois(b, e) * pois(s, e + m)
    )
    assert log_likelihood_day(b, s, a, d, m, e) == pytest.approx(math.log(naive), rel=1e-10)


def test_log_likelihood_no_underflow_at_large_counts() -> None:
    """B*log(1+mu/eps) would overflow naive (1+mu/eps)^B — log space survives."""
    ll = log_likelihood_day(5000.0, 3000.0, 0.5, 0.4, 400.0, 100.0)
    assert math.isfinite(ll)


def test_simulate_pin_days_dgp_moments() -> None:
    """E[B] = eps + alpha(1-delta)mu; E[S] = eps + alpha*delta*mu."""
    days = simulate_pin_days(alpha=0.4, delta=0.3, mu=100.0, eps=50.0, n_days=5000, seed=1)
    exp_b = 50.0 + 0.4 * 0.7 * 100.0  # 78
    exp_s = 50.0 + 0.4 * 0.3 * 100.0  # 62
    assert np.mean(days[:, 0]) == pytest.approx(exp_b, rel=0.03)
    assert np.mean(days[:, 1]) == pytest.approx(exp_s, rel=0.03)


def test_fit_pin_recovers_ground_truth() -> None:
    days = simulate_pin_days(alpha=0.35, delta=0.3, mu=80.0, eps=50.0, n_days=400, seed=2)
    fit = fit_pin(days)
    assert fit.pin == pytest.approx(pin_of(0.35, 0.3, 80.0, 50.0), abs=0.08)
    assert fit.alpha == pytest.approx(0.35, abs=0.12)
    assert fit.loglik < 0
    assert fit.n_days == 400


def test_fit_pin_no_informed_flow() -> None:
    """Honest negative control: pure ZI days must yield PIN ~= 0."""
    days = simulate_pin_days(alpha=0.0, delta=0.5, mu=80.0, eps=50.0, n_days=300, seed=3)
    fit = fit_pin(days)
    assert fit.pin < 0.05


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        simulate_pin_days(alpha=1.5, delta=0.3, mu=80.0, eps=50.0, n_days=10)
    with pytest.raises(ValueError):
        simulate_pin_days(alpha=0.3, delta=0.3, mu=80.0, eps=0.0, n_days=10)
    with pytest.raises(ValueError):
        simulate_pin_days(alpha=0.3, delta=0.3, mu=80.0, eps=50.0, n_days=0)
    with pytest.raises(ValueError):
        log_likelihood_day(10.0, 5.0, 0.3, 0.3, 80.0, -1.0)
    with pytest.raises(ValueError):
        log_likelihood_day(-1.0, 5.0, 0.3, 0.3, 80.0, 50.0)
    with pytest.raises(ValueError):
        fit_pin(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        fit_pin(np.full((50, 2), np.nan))


def test_bench_schema_and_claims() -> None:
    r = pin_estimator_bench(n_days=200, n_boot=50, seed=0)
    assert r["schema"] == PIN_SCHEMA
    assert r["kind"] == "pin_estimator"
    assert r["data_label"] == "SYNTHETIC"
    assert r["research_only"] is True
    assert r["informed_recovery_ok"] is True
    assert r["pure_zi_pin_small"] is True
    assert len(r["payload_sha256"]) == 64


def test_bench_determinism() -> None:
    a = pin_estimator_bench(n_days=150, n_boot=50, seed=7)
    b = pin_estimator_bench(n_days=150, n_boot=50, seed=7)
    assert a["payload_sha256"] == b["payload_sha256"]
