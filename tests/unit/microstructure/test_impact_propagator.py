"""Tests for microstructure/impact_propagator.py — response-function lane."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.impact_propagator import (
    IMPACT_PROP_SCHEMA,
    ResponseCurve,
    fit_power_law,
    impact_propagator_bench,
    mo_response_series,
    sign_shuffled_control,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config


def _synthetic_curve() -> ResponseCurve:
    # R(k) = 2 k^{-0.5}, counts sufficient for the fit
    lags = np.arange(0, 21, dtype=float)
    resp = np.zeros_like(lags)
    resp[1:] = 2.0 * lags[1:] ** -0.5
    counts = np.full(lags.shape, 100, dtype=np.int64)
    return ResponseCurve(lags=lags, response=resp, n_events=200, counts=counts)


def test_fit_power_law_recovers_theta() -> None:
    fit = fit_power_law(_synthetic_curve())
    assert fit["theta"] == pytest.approx(0.5, abs=0.02)
    assert fit["a"] == pytest.approx(2.0, rel=0.05)
    assert fit["r2"] > 0.99


def test_fit_power_law_fail_closed() -> None:
    with pytest.raises(TypeError):
        fit_power_law(object())
    # too few usable lags
    lags = np.array([0.0, 1.0, 2.0])
    resp = np.array([0.0, 1.0, 0.5])
    counts = np.full(3, 2, dtype=np.int64)
    with pytest.raises(ValueError):
        fit_power_law(ResponseCurve(lags=lags, response=resp, n_events=10, counts=counts))


def test_mo_response_series_smoke() -> None:
    cfg = santa_fe_config(seed=2)
    curve = mo_response_series(cfg, horizon=400.0, max_lag=20)
    assert curve.n_events > 0
    assert curve.lags.shape == (21,)
    assert np.isfinite(curve.response[0])
    assert np.all(curve.counts >= 0)
    assert np.all(np.diff(curve.counts[curve.counts > 0]) <= 0)  # counts shrink with lag


def test_sign_shuffle_is_flat() -> None:
    cfg = santa_fe_config(seed=2)
    ctrl = sign_shuffled_control(cfg, horizon=400.0, max_lag=10, seed=0)
    finite = ctrl.response[np.isfinite(ctrl.response)]
    # shuffled signs: response magnitude small vs the true curve's R(0)
    assert np.abs(finite).max() < 5.0  # tick units; loose bound


def test_response_fail_closed() -> None:
    cfg = santa_fe_config(seed=2)
    with pytest.raises(ValueError):
        mo_response_series(cfg, horizon=0.0)
    with pytest.raises(ValueError):
        mo_response_series(cfg, horizon=10.0, max_lag=0)
    with pytest.raises(ValueError):
        sign_shuffled_control(cfg, horizon=10.0, seed=-1)


def test_bench_receipt_schema_and_determinism() -> None:
    r1 = impact_propagator_bench(horizon=800.0, max_lag=20, seed=5)
    r2 = impact_propagator_bench(horizon=800.0, max_lag=20, seed=5)
    assert r1["schema"] == IMPACT_PROP_SCHEMA
    assert r1["kind"] == "impact_propagator"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    assert r1["n_mo_events"] >= 30
    assert len(r1["response_ticks"]) == 21
    assert r1["control_max_abs_ticks"] < 5.0
    assert len(r1["payload_sha256"]) == 64
    assert r1["payload_sha256"] == r2["payload_sha256"]


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        impact_propagator_bench(horizon=-1.0)
