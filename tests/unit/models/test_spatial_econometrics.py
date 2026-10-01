"""Tests for spatial autoregressive regression (models/spatial_econometrics.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.spatial_econometrics import (
    bench_spatial_econometrics,
    lattice_weights,
    row_standardize,
    sar_fit,
    synth_sar,
)


def _panel(**kw):
    return synth_sar(seed=27, **kw)


def test_rho_recovery():
    d = _panel(rho=0.6)
    out = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    assert abs(float(out["rho"]) - 0.6) < 0.15


def test_beta_beats_ols():
    d = _panel(rho=0.6)
    out = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    assert abs(float(out["beta_1"]) - 1.0) < abs(float(out["beta_ols_1"]) - 1.0)


def test_null_rho_near_zero():
    d = _panel(rho=0.0)
    out = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    assert abs(float(out["rho"])) < 0.25


def test_weights_row_standardized():
    w = lattice_weights(5)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(np.diag(w) == 0.0)


def test_moran_clears():
    d = _panel(rho=0.6)
    out = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    assert abs(float(out["moran_z"])) < 2.0


def test_validation():
    d = _panel()
    y, x, w = map(np.asarray, (d["y"], d["x"], d["w"]))
    with pytest.raises(ValueError):
        sar_fit(y[:5], x[:5], w[:5, :5])
    with pytest.raises(ValueError):
        sar_fit(y, x, w[:-1, :-1])
    with pytest.raises(ValueError):
        row_standardize(np.zeros((6, 6)))
    y2 = y.copy()
    y2[3] = np.nan
    with pytest.raises(ValueError):
        sar_fit(y2, x, w)


def test_determinism():
    d = _panel()
    a = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    b = sar_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["w"]))
    assert float(a["rho"]) == float(b["rho"])


def test_bench_keys():
    out = bench_spatial_econometrics()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
