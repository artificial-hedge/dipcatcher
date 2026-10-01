"""Tests for DiD diagnostics (models/did_diagnostics.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.did_diagnostics import (
    bacon_decomposition,
    bench_did_diagnostics,
    sun_abraham_effects,
    synth_staggered,
    twfe_beta,
)


def _panel(**kw):
    return synth_staggered(seed=13, **kw)


def test_twfe_positive_under_effect():
    d = _panel(tau_step=0.5)
    b = twfe_beta(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    assert b > 0.5


def test_null_twfe_small():
    d = _panel(tau_step=0.0)
    b = twfe_beta(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    assert abs(b) < 0.5


def test_bacon_components_and_weights():
    d = _panel()
    bd = bacon_decomposition(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    w = np.asarray(bd["weights"])
    assert abs(w.sum() - 1.0) < 1e-8
    assert bd["n_components"] > 2
    kk = np.asarray(bd["kinds"])
    assert (kk < 0.5).any()  # forbidden comparisons present
    assert 0.0 < bd["forbidden_weight"] < 1.0


def test_sun_abraham_dynamic_effects():
    d = _panel(tau_step=0.6)
    sa = sun_abraham_effects(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    rp = np.asarray(sa["rel_periods"])
    catt = np.asarray(sa["catt"])
    assert 0 in rp
    post = catt[rp >= 0]
    # effects should grow with relative period (τ_k = 0.6·(k+1))
    assert post.size >= 2
    assert post[-1] > post[0]


def test_forbidden_weight_grows_with_heterogeneity_flag():
    d = _panel(tau_step=0.8)
    bd = bacon_decomposition(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    assert bd["forbidden_weight"] > 0.1


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    u = np.asarray(d["unit"])
    t = np.asarray(d["time"])
    g = np.asarray(d["g"])
    with pytest.raises(ValueError):
        bacon_decomposition(y[:4], u, t, g)
    with pytest.raises(ValueError):
        bacon_decomposition(y, u, t, np.zeros_like(g))  # no treated
    with pytest.raises(ValueError):
        bacon_decomposition(y, u, t, np.ones_like(g))  # no never-treated
    with pytest.raises(ValueError):
        sun_abraham_effects(y, u, t, np.ones_like(g))


def test_determinism():
    d = _panel()
    a = bacon_decomposition(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    b = bacon_decomposition(
        np.asarray(d["y"]), np.asarray(d["unit"]), np.asarray(d["time"]), np.asarray(d["g"])
    )
    assert a["beta_bacon"] == b["beta_bacon"]


def test_bench_keys():
    out = bench_did_diagnostics()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_forbidden_w"] > 0.0
