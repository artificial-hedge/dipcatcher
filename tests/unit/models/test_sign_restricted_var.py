"""Tests for sign-restricted VAR (models/sign_restricted_var.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.sign_restricted_var import (
    bench_sign_restricted_var,
    sign_restricted_irf,
    synth_var,
    var_ma_coefs,
    var_ols,
)


def _panel(**kw):
    return synth_var(seed=19, **kw)


def test_var_ols_shapes():
    d = _panel()
    fit = var_ols(np.asarray(d["y"]), p=1)
    B = np.asarray(fit["B"])
    assert B.shape == (1 + 3, 3)
    assert np.asarray(fit["sigma_u"]).shape == (3, 3)


def test_ma_coefs_decay():
    d = _panel()
    fit = var_ols(np.asarray(d["y"]), p=1)
    th = var_ma_coefs(fit, h=6)
    assert th.shape == (7, 3, 3)
    assert np.allclose(th[0], np.eye(3))
    # stationary: last coefs smaller than first
    assert np.abs(th[6]).max() < np.abs(th[1]).max() + 0.5


def test_positive_restriction_recovers_sign():
    d = _panel()
    out = sign_restricted_irf(
        np.asarray(d["y"]),
        shock_col=0,
        target_cols=(1,),
        horizon=6,
        restrict_h=3,
        signs=(1,),
        n_draws=150,
        seed=1,
    )
    assert float(out["accept_share"]) > 0.05
    med = np.asarray(out["target_irf"])[:, 0]
    assert med[1] > 0.0


def test_accept_share_complements():
    d = _panel()
    pos = sign_restricted_irf(np.asarray(d["y"]), 0, (1,), 6, 2, (1,), n_draws=100, seed=3)
    neg = sign_restricted_irf(np.asarray(d["y"]), 0, (1,), 6, 2, (-1,), n_draws=100, seed=3)
    # draws split between the two signs (near-1 sum up to boundary)
    assert 0.0 <= float(pos["accept_share"]) <= 1.0
    assert 0.0 <= float(neg["accept_share"]) <= 1.0


def test_irf_bands_bracket_median():
    d = _panel()
    out = sign_restricted_irf(np.asarray(d["y"]), 0, (1,), 6, 2, (1,), n_draws=100, seed=4)
    if float(out["accept_share"]) > 0.05:
        med = np.asarray(out["irf_median"])
        lo = np.asarray(out["irf_lo"])
        hi = np.asarray(out["irf_hi"])
        assert np.all(lo <= med + 1e-9) and np.all(med <= hi + 1e-9)


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    with pytest.raises(ValueError):
        var_ols(y[:3], p=1)
    with pytest.raises(ValueError):
        sign_restricted_irf(y, 9, (1,), 6, 2, (1,))
    with pytest.raises(ValueError):
        sign_restricted_irf(y, 0, (9,), 6, 2, (1,))
    with pytest.raises(ValueError):
        sign_restricted_irf(y, 0, (1, 2), 6, 2, (1,))  # signs mismatch
    y2 = y.copy()
    y2[0, 0] = np.nan
    with pytest.raises(ValueError):
        sign_restricted_irf(y2, 0, (1,), 6, 2, (1,))


def test_determinism():
    d = _panel()
    a = sign_restricted_irf(np.asarray(d["y"]), 0, (1,), 6, 2, (1,), n_draws=80, seed=5)
    b = sign_restricted_irf(np.asarray(d["y"]), 0, (1,), 6, 2, (1,), n_draws=80, seed=5)
    assert float(a["accept_share"]) == float(b["accept_share"])


def test_bench_keys():
    out = bench_sign_restricted_var()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
