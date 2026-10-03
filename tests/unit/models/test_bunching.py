"""Tests for bunching estimation at kinks (models/bunching.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.bunching import (
    bench_bunching,
    bunching_estimate,
    excess_mass,
    synth_bunching,
)


def test_b_at_kink_positive():
    d = synth_bunching(n=8000, strength=0.5, seed=9)
    est = bunching_estimate(
        np.asarray(d["x"]), kink=0.0, binwidth=0.05, window=0.15, poly_deg=5, n_boot=40, seed=9
    )
    assert est["b"] > 0.5
    assert est["b"] / est["se"] > 3.0


def test_no_bunching_clean():
    d = synth_bunching(n=8000, strength=0.0, seed=10)
    est = bunching_estimate(
        np.asarray(d["x"]), kink=0.0, binwidth=0.05, window=0.15, poly_deg=5, n_boot=40, seed=10
    )
    assert abs(est["b"]) < 2.0


def test_b_scales_with_strength():
    weak = synth_bunching(n=8000, strength=0.2, seed=11)
    strong = synth_bunching(n=8000, strength=0.6, seed=11)
    bw = bunching_estimate(
        np.asarray(weak["x"]), 0.0, binwidth=0.05, window=0.15, n_boot=30, seed=11
    )["b"]
    bs = bunching_estimate(
        np.asarray(strong["x"]), 0.0, binwidth=0.05, window=0.15, n_boot=30, seed=11
    )["b"]
    assert bs > bw


def test_excess_mass_ci():
    d = synth_bunching(n=8000, strength=0.5, seed=12)
    em = excess_mass(np.asarray(d["x"]), kink=0.0, binwidth=0.05, window=0.15)
    assert em["ci_lb"] < em["b"] < em["ci_ub"]
    assert em["se"] > 0


def test_synth_piles_at_kink():
    d = synth_bunching(n=8000, strength=0.6, seed=13)
    x = np.asarray(d["x"])
    near = np.mean(np.abs(x) < 0.05)
    assert near > 0.08  # pile-up above the ~4% base share


def test_validation():
    with pytest.raises(ValueError):
        bunching_estimate(np.ones(200), kink=0.0)
    with pytest.raises(ValueError):
        bunching_estimate(np.random.default_rng(0).normal(size=1000), kink=10.0)
    with pytest.raises(ValueError):
        bunching_estimate(np.random.default_rng(0).normal(size=1000), kink=0.0, window=1e-6)


def test_determinism():
    d = synth_bunching(n=6000, strength=0.4, seed=14)
    a = bunching_estimate(np.asarray(d["x"]), 0.0, binwidth=0.05, window=0.15, n_boot=30, seed=14)[
        "b"
    ]
    b = bunching_estimate(np.asarray(d["x"]), 0.0, binwidth=0.05, window=0.15, n_boot=30, seed=14)[
        "b"
    ]
    assert a == b


def test_bench_keys():
    out = bench_bunching()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_b_z"] > 4.0
    assert out["synthetic_detects_kink"] == 1.0
    assert out["synthetic_determinism"] == 1.0
