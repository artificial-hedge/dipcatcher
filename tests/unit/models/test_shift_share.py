"""Tests for shift-share / Bartik IV (models/shift_share.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.shift_share import (
    bartik_instrument,
    bench_shift_share,
    rotemberg_weights,
    shift_share_iv,
    synth_shift_share,
)


def _panel(**kw):
    return synth_shift_share(seed=11, **kw)


def test_instrument_is_share_weighted():
    s = np.array([[0.5, 0.5], [0.25, 0.75]])
    g = np.array([2.0, 4.0])
    z = bartik_instrument(s, g)
    assert np.allclose(z, [3.0, 3.5])


def test_iv_recovers_beta():
    d = _panel(beta=1.0)
    iv = shift_share_iv(
        np.asarray(d["y"]),
        np.asarray(d["x_endog"]),
        np.asarray(d["shares"]),
        np.asarray(d["shocks"]),
    )
    assert abs(iv["beta_iv"] - 1.0) < 0.4
    assert iv["first_stage_f"] > 5.0


def test_iv_beats_ols_under_endogeneity():
    d = _panel(beta=1.0, endog=0.6)
    iv = shift_share_iv(
        np.asarray(d["y"]),
        np.asarray(d["x_endog"]),
        np.asarray(d["shares"]),
        np.asarray(d["shocks"]),
    )
    x = np.asarray(d["x_endog"])
    y = np.asarray(d["y"])
    ols = float(np.cov(x, y)[0, 1] / np.var(x))
    assert abs(iv["beta_iv"] - 1.0) < abs(ols - 1.0)


def test_null_beta_small():
    d = _panel(beta=0.0)
    iv = shift_share_iv(
        np.asarray(d["y"]),
        np.asarray(d["x_endog"]),
        np.asarray(d["shares"]),
        np.asarray(d["shocks"]),
    )
    assert abs(iv["beta_iv"]) < 0.6


def test_rotemberg_weights_sum():
    d = _panel()
    rw = rotemberg_weights(
        np.asarray(d["shares"]), np.asarray(d["shocks"]), np.asarray(d["x_endog"])
    )
    w = np.asarray(rw["weights"])
    assert abs(w.sum() - 1.0) < 1e-8
    assert 0.0 < rw["hhi"] <= 1.0


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    x = np.asarray(d["x_endog"])
    s = np.asarray(d["shares"])
    g = np.asarray(d["shocks"])
    with pytest.raises(ValueError):
        bartik_instrument(s[:, :5], g)  # column mismatch
    with pytest.raises(ValueError):
        bartik_instrument(-s, g)  # negative shares
    with pytest.raises(ValueError):
        bartik_instrument(s * 2.0, g)  # rows sum > 1
    with pytest.raises(ValueError):
        shift_share_iv(np.ones(5), x, s, g)
    with pytest.raises(ValueError):
        shift_share_iv(y, x[:4], s, g)
    with pytest.raises(ValueError):
        rotemberg_weights(s, g, x[:5])


def test_determinism():
    d = _panel()
    a = shift_share_iv(
        np.asarray(d["y"]),
        np.asarray(d["x_endog"]),
        np.asarray(d["shares"]),
        np.asarray(d["shocks"]),
    )
    b = shift_share_iv(
        np.asarray(d["y"]),
        np.asarray(d["x_endog"]),
        np.asarray(d["shares"]),
        np.asarray(d["shocks"]),
    )
    assert a["beta_iv"] == b["beta_iv"]


def test_bench_keys():
    out = bench_shift_share()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_beats_ols"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
