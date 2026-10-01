"""Tests for Newey-Powell series IV (models/nonparametric_iv.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.nonparametric_iv import (
    bench_nonparametric_iv,
    npiv_fit,
    synth_endogenous,
)


def test_dwh_detects_endogeneity():
    d = synth_endogenous(confound=0.7, seed=37)
    out = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert abs(float(out["dwh_t"])) > 2.0


def test_dwh_clean_under_exogeneity():
    d = synth_endogenous(confound=0.0, seed=37)
    out = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert abs(float(out["dwh_t"])) < 3.0


def test_first_stage_relevance():
    d = synth_endogenous(seed=37)
    out = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert float(out["first_stage_r2"]) > 0.3


def test_iv_differs_from_ols_under_confounding():
    d = synth_endogenous(confound=0.7, seed=37)
    out = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert float(out["iv_ols_gap"]) > 0.05


def test_validation():
    d = synth_endogenous(seed=37)
    y, x, z = np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"])
    with pytest.raises(ValueError):
        npiv_fit(y[:50], x[:50], z[:50])
    with pytest.raises(ValueError):
        npiv_fit(y, x * 0.0, z)  # constant x
    with pytest.raises(ValueError):
        npiv_fit(y, x, z * 0.0)  # constant z
    with pytest.raises(ValueError):
        npiv_fit(y, x, z, n_basis=2)
    weak = np.random.default_rng(3).normal(0, 1e-6, y.size)
    with pytest.raises(ValueError):
        npiv_fit(y, x, weak)  # instrument too weak


def test_determinism():
    d = synth_endogenous(seed=37)
    a = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    b = npiv_fit(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert float(a["dwh_t"]) == float(b["dwh_t"])
    assert float(a["iv_ols_gap"]) == float(b["iv_ols_gap"])


def test_bench_keys():
    out = bench_nonparametric_iv()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
