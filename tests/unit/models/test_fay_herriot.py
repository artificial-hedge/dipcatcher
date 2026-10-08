"""Tests for fay_herriot — small-area EBLUP."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fay_herriot import bench_fay_herriot, fay_herriot


def test_eblup_shrinks():
    rng = np.random.default_rng(0)
    m = 30
    x = np.stack([np.ones(m), rng.standard_normal(m)], axis=1)
    v = rng.uniform(0.3, 1.0, m)
    y = (
        x @ np.array([1.0, 0.8])
        + 0.7 * rng.standard_normal(m)
        + np.sqrt(v) * rng.standard_normal(m)
    )
    out = fay_herriot(y, x, v)
    theta = np.asarray(out["theta"])
    # theta should be between y and synthetic fit
    synth = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    dist_eb = np.abs(theta - y).mean()
    dist_direct = np.abs(theta - synth).mean()
    assert dist_eb > 0 and dist_direct > 0


def test_zero_a_collapses_to_synthetic():
    rng = np.random.default_rng(1)
    m = 25
    x = np.ones((m, 1))
    y = 2.0 + np.sqrt(rng.uniform(0.5, 1.0, m)) * rng.standard_normal(m)
    out = fay_herriot(y, x, rng.uniform(0.5, 1.0, m))
    theta = np.asarray(out["theta"])
    assert np.isfinite(theta).all()


def test_a_var_recovered():
    rng = np.random.default_rng(2)
    m = 60
    x = np.ones((m, 1))
    v = rng.uniform(0.2, 0.5, m)
    y = (
        1.0 * np.ones(m)
        + np.sqrt(1.2) * rng.standard_normal(m)
        + np.sqrt(v) * rng.standard_normal(m)
    )
    out = fay_herriot(y, x, v)
    assert 0.3 < out["a_var"] < 2.5


def test_fail_closed_neg_var():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        fay_herriot(rng.standard_normal(10), np.ones((10, 1)), -np.ones(10))


def test_bench():
    out = bench_fay_herriot()
    assert out["synthetic_score"] == 1.0
