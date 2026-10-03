"""Serial-correlation diagnostics (DW, h, BG, LB)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.serial_diagnostics import (
    breusch_godfrey,
    durbin_h,
    durbin_watson,
    ljung_box,
)


def _ar(seed=0, n=200, rho=0.0):
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, 1)
    return e


def test_dw_near_two_for_iid():
    out = durbin_watson(_ar(rho=0.0))
    assert 1.6 < out["stat"] < 2.4
    assert out["pvalue"] > 0.05


def test_dw_low_for_ar1():
    out = durbin_watson(_ar(rho=0.8))
    assert out["stat"] < 0.6
    assert out["pvalue"] < 0.01


def test_durbin_h():
    e = _ar(rho=0.7)
    out = durbin_h(e, var_b_lag=0.001)
    assert out["pvalue"] < 0.01
    assert out["used_m"] is False
    # when variance term blows up, falls back to m-test
    out2 = durbin_h(e, var_b_lag=0.1)
    assert out2["used_m"] is True


def test_bg_detects_ar2():
    e = _ar(rho=0.5)
    x = np.random.default_rng(1).normal(0, 1, e.size)[:, None]
    out = breusch_godfrey(e, x, lags=2)
    assert out["pvalue"] < 0.01
    assert out["df"] == 2


def test_bg_accepts_iid():
    e = _ar(rho=0.0)
    x = np.random.default_rng(2).normal(0, 1, e.size)[:, None]
    out = breusch_godfrey(e, x, lags=2)
    assert out["pvalue"] > 0.05


def test_ljung_box():
    e = _ar(rho=0.6)
    out = ljung_box(e, m=10)
    assert out["stat"] > 30
    assert out["pvalue"] < 0.001
    ok = ljung_box(_ar(rho=0.0), m=10)
    assert ok["pvalue"] > 0.05


def test_validation():
    with pytest.raises(ValueError):
        durbin_watson(np.ones(10))
    with pytest.raises(ValueError):
        ljung_box(np.full(30, np.nan))
