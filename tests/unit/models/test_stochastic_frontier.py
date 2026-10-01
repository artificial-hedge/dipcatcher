"""Tests for stochastic frontier analysis (models/stochastic_frontier.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.stochastic_frontier import (
    bench_stochastic_frontier,
    sfa_fit,
    synth_sfa,
)


def _panel(**kw):
    return synth_sfa(seed=25, **kw)


def test_beta_recovery():
    d = _panel()
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(np.asarray(out["beta"])[1]) - 0.7) < 0.15


def test_sigma_u_detected():
    d = _panel(sigma_u=0.6)
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["sigma_u"]) > 0.25


def test_pure_noise_smaller_su_than_strong():
    # σu→0 is a weakly-identified boundary case: honest check is that
    # strong inefficiency estimates MORE σu than near-symmetric noise.
    d_lo = _panel(sigma_u=0.05, sigma_v=0.5)
    d_hi = _panel(sigma_u=0.8, sigma_v=0.3)
    out_lo = sfa_fit(np.asarray(d_lo["y"]), np.asarray(d_lo["x"]))
    out_hi = sfa_fit(np.asarray(d_hi["y"]), np.asarray(d_hi["x"]))
    assert float(out_hi["sigma_u"]) > float(out_lo["sigma_u"]) + 0.15


def test_efficiency_range():
    d = _panel()
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    eff = np.asarray(out["efficiency"])
    assert np.all(eff > 0.0) and np.all(eff <= 1.0 + 1e-9)


def test_efficiency_tracks_u():
    d = _panel()
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    corr = float(np.corrcoef(np.asarray(out["efficiency"]), -np.asarray(d["u_true"]))[0, 1])
    assert corr > 0.4


def test_intercept_shifts_down():
    # frontier intercept should sit above mean residual (u subtracted)
    d = _panel()
    out = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(np.asarray(out["beta"])[0]) > -0.2


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    with pytest.raises(ValueError):
        sfa_fit(y[:6], x[:6])
    with pytest.raises(ValueError):
        sfa_fit(y[:5], x)
    x2 = x.copy()
    x2[0, 0] = np.nan
    with pytest.raises(ValueError):
        sfa_fit(y, x2)


def test_determinism():
    d = _panel()
    a = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = sfa_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(np.asarray(a["beta"])[1]) == float(np.asarray(b["beta"])[1])


def test_bench_keys():
    out = bench_stochastic_frontier()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
