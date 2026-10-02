"""Tests for watson — Watson U^2 circular uniformity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.watson import bench_watson, watson_u2


def test_accepts_uniform():
    rng = np.random.default_rng(0)
    th = rng.uniform(0.0, 2.0 * math.pi, size=200)
    out = watson_u2(th)
    assert out["p_u2"] > 0.05
    assert out["u2"] >= 0


def test_rejects_von_mises():
    rng = np.random.default_rng(1)
    th = rng.vonmises(0.0, 4.0, size=200)
    out = watson_u2(th)
    assert out["p_u2"] < 0.05
    assert out["rayleigh_p"] < 0.05


def test_rayleigh_uniform_large_p():
    rng = np.random.default_rng(2)
    th = rng.uniform(0.0, 2.0 * math.pi, size=300)
    out = watson_u2(th)
    assert out["rayleigh_r"] < 0.15


def test_wrapped_angles():
    rng = np.random.default_rng(3)
    th = rng.uniform(-math.pi, math.pi, size=150)
    out = watson_u2(th)
    assert out["p_u2"] > 0.01


def test_bad_inputs():
    with pytest.raises(ValueError):
        watson_u2(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        watson_u2(np.array([0.1] * 10 + [float("nan")] * 10))


def test_bench():
    assert bench_watson()["score"] == 1.0
