"""Tests for Arellano-Bond dynamic panel GMM (models/arellano_bond.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.arellano_bond import (
    arellano_bond,
    bench_arellano_bond,
    synth_dynamic_panel,
)


def _panel(**kw):
    return synth_dynamic_panel(seed=21, **kw)


def test_rho_recovery():
    d = _panel(rho=0.6)
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert abs(out["rho"] - 0.6) < 0.25


def test_beta_recovery():
    d = _panel(rho=0.6, beta=0.8)
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert abs(float(np.asarray(out["beta"])[0]) - 0.8) < 0.3


def test_ar2_absent_in_clean_dgp():
    d = _panel()
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert float(out["ar2_p"]) > 0.01


def test_sargan_runs():
    d = _panel()
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert float(out["sargan_j"]) >= 0.0
    assert 0.0 <= float(out["sargan_p"]) <= 1.0


def test_null_rho_small():
    d = _panel(rho=0.0)
    out = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert abs(out["rho"]) < 0.3


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    x = np.asarray(d["x"])
    g = np.asarray(d["groups"])
    with pytest.raises(ValueError):
        arellano_bond(y[:5], x, g)
    with pytest.raises(ValueError):
        arellano_bond(y, x[:4], g)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        arellano_bond(y2, x, g)


def test_determinism():
    d = _panel()
    a = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    b = arellano_bond(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["groups"]))
    assert a["rho"] == b["rho"]


def test_bench_keys():
    out = bench_arellano_bond()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
