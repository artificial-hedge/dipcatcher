"""Tests for interrater — agreement coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.interrater import (
    bench_interrater,
    bland_altman,
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    lin_ccc,
    weighted_kappa,
)


def test_perfect_agreement():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 3, size=100).astype(float)
    out = cohen_kappa(r, r)
    assert out["kappa"] == pytest.approx(1.0)


def test_chance_agreement_low_kappa():
    rng = np.random.default_rng(1)
    n = 2000
    r1 = rng.integers(0, 4, size=n).astype(float)
    r2 = rng.integers(0, 4, size=n).astype(float)
    out = cohen_kappa(r1, r2)
    assert abs(out["kappa"]) < 0.1


def test_weighted_kappa_near_misses():
    # quadratic weighting: near-misses hurt less than big gaps
    r1 = np.array([0, 0, 1, 1, 2, 2] * 10, dtype=float)
    r2 = r1.copy()
    r2[::3] = np.clip(r2[::3] + 1, 0, 2)  # adjacent misses
    out = weighted_kappa(r1, r2, weights="quadratic")
    assert 0.5 < out["kappa_weighted"] < 1.0


def test_fleiss_perfect():
    counts = np.zeros((20, 3))
    counts[np.arange(20), np.arange(20) % 3] = 5.0  # all 5 raters agree
    out = fleiss_kappa(counts)
    assert out["kappa_fleiss"] == pytest.approx(1.0)


def test_krippendorff_identical():
    r = np.tile(np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), (3, 1))
    out = krippendorff_alpha(r)
    assert out["alpha"] > 0.95


def test_lin_ccc_identity():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    out = lin_ccc(x, x)
    assert out["ccc"] == pytest.approx(1.0)
    out2 = lin_ccc(x, x + 2.0)
    assert out2["ccc"] < 0.8  # shifted scale hurts concordance


def test_bland_altman_bias():
    rng = np.random.default_rng(4)
    x = rng.normal(size=100)
    y = x + 0.5 + rng.normal(scale=0.1, size=100)
    out = bland_altman(y, x)
    assert out["bias"] == pytest.approx(0.5, abs=0.05)
    assert out["loa_high"] > out["bias"] > out["loa_low"]


def test_fail_closed_single_category():
    with pytest.raises(ValueError):
        cohen_kappa(np.ones(10), np.ones(10))


def test_bench():
    out = bench_interrater()
    assert out["score"] == 1.0
