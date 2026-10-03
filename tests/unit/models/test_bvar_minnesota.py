"""Tests for Minnesota-prior BVAR (models/bvar_minnesota.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.bvar_minnesota import (
    bench_bvar_minnesota,
    bvar_estimate,
    bvar_shrinkage_profile,
    minnesota_dummies,
    synth_bvar,
)


def _panel(**kw):
    return synth_bvar(seed=22, **kw)


def test_dummies_shape_and_prior():
    Yd, Xd = minnesota_dummies(3, 1, lambda_=0.1)
    assert Yd.shape[0] == 1 + 9 and Xd.shape[1] == 4
    # own-lag row for eq0 carries prior mean at column 0 only
    own_rows = np.where(Yd[:, 0] != 0)[0]
    assert own_rows.size == 1
    assert Yd[own_rows[0], 0] > 0


def test_tighter_pulls_toward_prior():
    d = _panel()
    y = np.asarray(d["y"])
    tight = np.asarray(bvar_estimate(y, p=1, lambda_=0.01)["B"])[1, 0]
    loose = np.asarray(bvar_estimate(y, p=1, lambda_=2.0)["B"])[1, 0]
    assert tight > loose  # prior mean 1.0


def test_bvar_beats_ols_short_sample():
    # persistent series (prior near truth), short noisy sample
    d = _panel(persistence=0.9, n=35)
    y = np.asarray(d["y"])
    A_true = np.asarray(d["A"])
    k = y.shape[1]
    B = np.asarray(bvar_estimate(y, p=1, lambda_=0.15)["B"])
    Y = y[1:]
    X = np.column_stack([np.ones(y.shape[0] - 1), y[:-1]])
    Bo, *_ = np.linalg.lstsq(X, Y, rcond=None)
    err_b = np.abs(np.diag(B[1 : 1 + k].T) - np.diag(A_true)).mean()
    err_o = np.abs(np.diag(Bo[1 : 1 + k].T) - np.diag(A_true)).mean()
    assert err_b < err_o


def test_cross_lags_shrunk():
    d = _panel()
    y = np.asarray(d["y"])
    B = np.asarray(bvar_estimate(y, p=1, lambda_=0.15)["B"])
    k = y.shape[1]
    A = B[1 : 1 + k].T
    cross = np.abs(A - np.diag(np.diag(A))).mean()
    assert cross < 0.15


def test_shrinkage_profile_grid():
    d = _panel()
    out = bvar_shrinkage_profile(np.asarray(d["y"]), p=1)
    assert np.asarray(out["own_lag1"]).shape == (4, 3)


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    with pytest.raises(ValueError):
        bvar_estimate(y[:3], p=1)
    with pytest.raises(ValueError):
        bvar_estimate(y, p=1, lambda_=0.0)
    with pytest.raises(ValueError):
        minnesota_dummies(3, 1, sigma_scales=np.array([1.0, -1.0, 1.0]))
    y2 = y.copy()
    y2[0, 0] = np.nan
    with pytest.raises(ValueError):
        bvar_estimate(y2, p=1)


def test_determinism():
    d = _panel()
    a = bvar_estimate(np.asarray(d["y"]), p=1, lambda_=0.15)
    b = bvar_estimate(np.asarray(d["y"]), p=1, lambda_=0.15)
    assert np.array_equal(np.asarray(a["B"]), np.asarray(b["B"]))


def test_bench_keys():
    out = bench_bvar_minnesota()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
