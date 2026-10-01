"""Tests for BSTS causal impact (models/causal_impact.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.causal_impact import (
    bench_causal_impact,
    causal_impact,
    synth_causal_impact,
)


def test_recovers_effect_significantly():
    d = synth_causal_impact(n=120, seed=20261231 + 185)
    est = causal_impact(
        np.asarray(d["y"]), int(d["t_int"]), x=np.asarray(d["x"]), n_sims=600, seed=5
    )
    assert est["cum_effect_z"] > 2.0
    assert est["posterior_tail_prob"] < 0.1
    assert est["cum_effect"] > 0.0


def test_null_reports_no_effect():
    d = synth_causal_impact(n=120, effect=0.0, seed=6)
    est = causal_impact(
        np.asarray(d["y"]), int(d["t_int"]), x=np.asarray(d["x"]), n_sims=600, seed=6
    )
    # honest posterior: wrong-side mass should not collapse to ~0
    assert est["posterior_tail_prob"] > 0.01


def test_no_covariates():
    d = synth_causal_impact(n=100, seed=7)
    est = causal_impact(np.asarray(d["y"]), int(d["t_int"]), x=None, n_sims=400, seed=7)
    assert math.isfinite(est["cum_effect"])


def test_counterfactual_shape():
    d = synth_causal_impact(n=80, seed=8)
    t_int = int(d["t_int"])
    est = causal_impact(np.asarray(d["y"]), t_int, x=np.asarray(d["x"]), n_sims=200, seed=8)
    cf = np.asarray(est["counterfactual_mean"])
    sims = np.asarray(est["pointwise_effects_sims"])
    assert cf.shape == (len(d["y"]) - t_int,)
    assert sims.shape == (200, len(d["y"]) - t_int)


def test_validation():
    with pytest.raises(ValueError):
        causal_impact(np.ones(8), 4)
    with pytest.raises(ValueError):
        causal_impact(np.random.default_rng(0).normal(size=50), 2)
    with pytest.raises(ValueError):
        causal_impact(np.random.default_rng(0).normal(size=50), 49)
    with pytest.raises(ValueError):
        causal_impact(np.random.default_rng(0).normal(size=50), 30, x=np.ones((40, 2)))


def test_determinism():
    d = synth_causal_impact(n=80, seed=9)
    a = causal_impact(np.asarray(d["y"]), int(d["t_int"]), x=np.asarray(d["x"]), n_sims=300, seed=9)
    b = causal_impact(np.asarray(d["y"]), int(d["t_int"]), x=np.asarray(d["x"]), n_sims=300, seed=9)
    assert a["cum_effect"] == b["cum_effect"]


def test_larger_effect_larger_stat():
    small = synth_causal_impact(n=120, effect=0.8, seed=10)
    big = synth_causal_impact(n=120, effect=3.0, seed=10)
    e_s = causal_impact(
        np.asarray(small["y"]), int(small["t_int"]), x=np.asarray(small["x"]), n_sims=300, seed=10
    )
    e_b = causal_impact(
        np.asarray(big["y"]), int(big["t_int"]), x=np.asarray(big["x"]), n_sims=300, seed=10
    )
    assert e_b["cum_effect"] > e_s["cum_effect"]


def test_bench_keys():
    out = bench_causal_impact()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_cum_effect_err"] < out["synthetic_cum_effect"]
    assert out["synthetic_determinism"] == 1.0
