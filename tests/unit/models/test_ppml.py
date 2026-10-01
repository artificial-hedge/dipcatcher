"""Tests for PPML gravity estimation (models/ppml.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.ppml import bench_ppml, ppml_fit, synth_gravity


def test_beta_recovered():
    d = synth_gravity(beta_x=0.8, seed=37)
    out = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    # heteroskedastic-gamma DGP is noisy; PPML keeps β within ~.5
    assert abs(float(out["beta_x"]) - 0.8) < 0.45


def test_null_beta():
    d = synth_gravity(beta_x=0.0, seed=37)
    out = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_x"])) < 0.2


def test_overdispersion_detected():
    d = synth_gravity(hetero=0.9, seed=37)
    out = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["phi_overdispersion"]) > 1.5


def test_tstat_significant():
    d = synth_gravity(beta_x=0.8, seed=37)
    out = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["t_x"]) > 3.0


def test_validation():
    d = synth_gravity(seed=37)
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        ppml_fit(-np.abs(y), x)  # negatives
    with pytest.raises(ValueError):
        ppml_fit(y[:40], x[:40])
    with pytest.raises(ValueError):
        ppml_fit(y, x[:, :1] * 0.0)  # constant x
    with pytest.raises(ValueError):
        ppml_fit(y * 0.0, x)  # all zeros
    bad = y.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        ppml_fit(bad, x)


def test_determinism():
    d = synth_gravity(seed=37)
    a = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = ppml_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(a["beta_x"]) == float(b["beta_x"])


def test_bench_keys():
    out = bench_ppml()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
