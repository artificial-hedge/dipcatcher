"""Tests for sieve partial-linear estimation (models/sieve_estimation.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.sieve_estimation import (
    bench_sieve_estimation,
    sieve_partial_linear,
    synth_partial_linear,
)


def test_beta_recovered():
    d = synth_partial_linear(beta=0.8, seed=37)
    out, _ = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert abs(float(out["beta_0"]) - 0.8) < 0.15


def test_g_tracked():
    d = synth_partial_linear(seed=37)
    _, g_hat = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    g = np.asarray(d["g_true"]) - np.asarray(d["g_true"]).mean()
    assert np.corrcoef(g_hat, g)[0, 1] > 0.9


def test_beats_linear_fit():
    d = synth_partial_linear(seed=37)
    out, _ = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert float(out["r2_sieve"]) > float(out["r2_linear"]) + 0.2


def test_linear_g_still_ok():
    # when g is truly linear the sieve doesn't blow up
    rng = np.random.default_rng(37)
    z = rng.uniform(0, 4, 400)
    x = rng.normal(0, 1, 400)
    y = 0.8 * x + 0.5 * z + rng.normal(0, 0.2, 400)
    out, _ = sieve_partial_linear(y, x, z)
    assert abs(float(out["beta_0"]) - 0.8) < 0.15


def test_null_beta():
    d = synth_partial_linear(beta=0.0, seed=37)
    out, _ = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert abs(float(out["beta_0"])) < 0.15


def test_validation():
    d = synth_partial_linear(seed=37)
    y, x, z = np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"])
    with pytest.raises(ValueError):
        sieve_partial_linear(y[:40], x[:40], z[:40])
    with pytest.raises(ValueError):
        sieve_partial_linear(y, x, np.full(y.size, 1.0))
    with pytest.raises(ValueError):
        sieve_partial_linear(y, x, z, n_basis=25)
    z_nan = z.copy()
    z_nan[0] = np.nan
    with pytest.raises(ValueError):
        sieve_partial_linear(y, x, z_nan)


def test_determinism():
    d = synth_partial_linear(seed=37)
    a, _ = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    b, _ = sieve_partial_linear(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert float(a["beta_0"]) == float(b["beta_0"])


def test_bench_keys():
    out = bench_sieve_estimation()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
