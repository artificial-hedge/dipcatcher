"""Tests for matrix-completion causal panels (models/matrix_completion.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.matrix_completion import (
    bench_matrix_completion,
    mc_att,
    soft_impute,
    synth_mc_panel,
)


def test_soft_impute_recovers_missing():
    rng = np.random.default_rng(5)
    lam = rng.normal(0, 1, (25, 2))
    f = rng.normal(0, 1, (40, 2))
    y = lam @ f.T + rng.normal(0, 0.1, (25, 40))
    mask = np.ones_like(y, bool)
    mask[:5, -8:] = False
    out = soft_impute(y, mask)
    cf = np.asarray(out["completed"])
    err = np.abs((cf - (lam @ f.T))[:5, -8:]).mean()
    assert err < 0.5


def test_mc_att_recovers_tau():
    d = synth_mc_panel(seed=6, tau=1.2)
    est = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert abs(est["att"] - 1.2) < 0.4


def test_null_att_small():
    d = synth_mc_panel(seed=7, tau=0.0)
    est = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert abs(est["att"]) < 0.5


def test_completed_shape_and_weights():
    d = synth_mc_panel(seed=8)
    est = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    cf = np.asarray(est["counterfactual"])
    assert cf.shape == np.asarray(d["y"]).shape
    assert est["eff_rank"] >= 1.0


def test_placebo_rmse_positive():
    d = synth_mc_panel(seed=9)
    est = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert est["impute_rmse_control"] > 0.0


def test_validation():
    with pytest.raises(ValueError):
        mc_att(np.ones((3, 4)), np.array([1, 0, 0]), 2)
    with pytest.raises(ValueError):
        mc_att(np.random.default_rng(0).normal(size=(10, 20)), np.zeros(10), 10)
    with pytest.raises(ValueError):
        soft_impute(np.ones((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        soft_impute(np.ones((4, 4)), np.ones((4, 5)))


def test_determinism():
    d = synth_mc_panel(seed=10)
    a = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    b = mc_att(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert a["att"] == b["att"]


def test_bench_keys():
    out = bench_matrix_completion()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_att_err"] < 0.5
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
