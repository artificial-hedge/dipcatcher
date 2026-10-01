"""Tests for many/weak instrument estimators (models/many_iv.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.many_iv import (
    bench_many_iv,
    hful,
    jive,
    liml,
    synth_many_iv,
)


def _panel(**kw):
    return synth_many_iv(seed=16, pi_strength=0.10, k=40, n=400, **kw)


def test_liml_recovers_beta():
    d = _panel(beta=1.0)
    out = liml(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert abs(out["beta"] - 1.0) < 0.25


def test_liml_beats_2sls_weak():
    d = _panel(beta=1.0)
    y = np.asarray(d["y"])
    x = np.asarray(d["x_endog"])
    z = np.asarray(d["z"])
    Z = np.column_stack([np.ones(y.size), z])
    Px = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ x
    b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(y.size), Px]), y, rcond=None)
    out = liml(y, x, z)
    assert abs(out["beta"] - 1.0) < abs(float(b2[1]) - 1.0)


def test_jive_reasonable():
    d = _panel(beta=1.0)
    out = jive(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert abs(out["beta"] - 1.0) < 0.6


def test_hful_reasonable():
    d = _panel(beta=1.0)
    out = hful(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert abs(out["beta"] - 1.0) < 0.6
    assert out["mean_leverage"] > 0.0


def test_null_liml_small():
    d = _panel(beta=0.0)
    out = liml(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert abs(out["beta"]) < 0.5


def test_z_finite_and_signed():
    d = _panel(beta=1.0)
    out = liml(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert math.isfinite(out["z"])
    assert out["se"] > 0.0


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    x = np.asarray(d["x_endog"])
    z = np.asarray(d["z"])
    with pytest.raises(ValueError):
        liml(np.ones(3), x, z)
    with pytest.raises(ValueError):
        liml(y, np.column_stack([x, x]), z)  # kx != 1
    with pytest.raises(ValueError):
        jive(y, x[:5], z)
    with pytest.raises(ValueError):
        hful(y, x, z[:3])
    with pytest.raises(ValueError):
        liml(y, x, np.empty((y.size, 0)))  # zero instruments


def test_determinism():
    d = _panel()
    a = liml(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    b = liml(np.asarray(d["y"]), np.asarray(d["x_endog"]), np.asarray(d["z"]))
    assert a["beta"] == b["beta"]


def test_bench_keys():
    out = bench_many_iv()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_beats_2sls"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
