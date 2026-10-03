"""Tests for microstructure/hawkes_calibrate.py — exp-Hawkes MLE lane."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.hawkes_calibrate import (
    HAWKES_CAL_SCHEMA,
    fit_hawkes_exp,
    hawkes_cal_bench,
    hawkes_exp_loglik,
    hawkes_ogata,
    zi_mo_times,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config


def test_loglik_poisson_exact() -> None:
    # alpha=0 => homogeneous Poisson: logL = n log mu - mu*T
    t = np.array([1.0, 2.0, 3.0, 4.0])
    ll = hawkes_exp_loglik(t, mu=2.0, alpha=0.0, beta=5.0)
    assert ll == pytest.approx(4.0 * np.log(2.0) - 2.0 * 4.0)


def test_loglik_fail_closed() -> None:
    t = np.array([1.0, 2.0, 3.0])
    assert hawkes_exp_loglik(t, mu=0.0, alpha=0.1, beta=1.0) == float("-inf")
    assert hawkes_exp_loglik(t, mu=1.0, alpha=-0.1, beta=1.0) == float("-inf")
    with pytest.raises(ValueError):
        hawkes_exp_loglik(np.array([1.0]), 1.0, 0.1, 1.0)
    with pytest.raises(ValueError):
        hawkes_exp_loglik(np.array([2.0, 1.0, 3.0]), 1.0, 0.1, 1.0)
    with pytest.raises(ValueError):
        hawkes_exp_loglik(np.array([1.0, float("nan"), 3.0]), 1.0, 0.1, 1.0)


def test_ogata_determinism_and_shape() -> None:
    a = hawkes_ogata(0.5, 0.8, 4.0, 100.0, seed=1)
    b = hawkes_ogata(0.5, 0.8, 4.0, 100.0, seed=1)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a) > 0.0)
    assert np.all(a < 100.0)
    # self-exciting -> more events than Poisson with same base rate
    n_poisson_mean = 0.5 * 100.0 / (1 - 0.8 / 4.0)
    assert a.size > 0.5 * n_poisson_mean


def test_ogata_fail_closed() -> None:
    with pytest.raises(ValueError):
        hawkes_ogata(0.5, 4.0, 4.0, 10.0, seed=0)  # supercritical
    with pytest.raises(ValueError):
        hawkes_ogata(0.0, 0.8, 4.0, 10.0, seed=0)
    with pytest.raises(ValueError):
        hawkes_ogata(0.5, 0.8, 4.0, 0.0, seed=0)
    with pytest.raises(ValueError):
        hawkes_ogata(0.5, 0.8, 4.0, 10.0, seed=-1)


def test_fit_recovers_hawkes_params() -> None:
    t = hawkes_ogata(0.6, 0.9, 3.0, 600.0, seed=4)
    fit = fit_hawkes_exp(t)
    assert fit.stationary
    assert fit.mu == pytest.approx(0.6, rel=0.35)
    assert fit.alpha == pytest.approx(0.9, rel=0.5)
    assert fit.beta == pytest.approx(3.0, rel=0.6)
    assert fit.branching < 1.0
    assert fit.n_events == t.size


def test_fit_poisson_low_branching() -> None:
    # pure Poisson stream -> fitted branching should be near zero
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.exponential(1.0 / 0.5, 400))
    fit = fit_hawkes_exp(t)
    assert fit.branching < 0.35
    assert fit.mu == pytest.approx(0.5, rel=0.35)


def test_fit_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_hawkes_exp(np.array([1.0, 2.0]))


def test_zi_mo_times() -> None:
    t = zi_mo_times(santa_fe_config(seed=3), 200.0)
    assert t.size > 0
    assert np.all(np.diff(t) > 0.0)


def test_bench_receipt_schema_and_determinism() -> None:
    r1 = hawkes_cal_bench(horizon=300.0, seed=8, zi_horizon=400.0)
    r2 = hawkes_cal_bench(horizon=300.0, seed=8, zi_horizon=400.0)
    assert r1["schema"] == HAWKES_CAL_SCHEMA
    assert r1["kind"] == "hawkes_cal"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    rec = r1["recovery"]
    assert rec["mu_rel_err"] < 0.6
    assert rec["branching_hat"] < 1.0
    zi = r1["zi_negative_control"]
    assert zi["n_mo_events"] > 0
    assert zi["branching_hat"] < 0.5  # Poisson-ish stream: little excitation
    assert len(r1["payload_sha256"]) == 64
    assert r1["payload_sha256"] == r2["payload_sha256"]


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        hawkes_cal_bench(horizon=0.0)
    with pytest.raises(ValueError):
        hawkes_cal_bench(zi_horizon=-1.0)
