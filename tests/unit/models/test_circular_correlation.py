"""Circular correlation tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.circular_correlation import (
    bench_circular_correlation,
    fisher_lee_circular,
    jammalamadaka_sarma,
    mardia_circular_linear,
)


def test_mardia_dependent_pair():
    rng = np.random.default_rng(0)
    th = rng.vonmises(0.0, 1.0, 150)
    x = 4.0 * np.cos(th) + rng.normal(0, 0.3, 150)
    out = mardia_circular_linear(th, x, n_perm=200, seed=0)
    assert out["r2"] > 0.7
    assert out["p"] < 0.05


def test_mardia_independent_pair():
    rng = np.random.default_rng(1)
    th = rng.vonmises(0.0, 1.0, 150)
    x = rng.normal(0, 1, 150)
    out = mardia_circular_linear(th, x, n_perm=200, seed=0)
    assert out["p"] > 0.05


def test_fisher_lee_coupled():
    rng = np.random.default_rng(2)
    a = rng.vonmises(0.3, 1.0, 100)
    b = np.mod(a + rng.normal(0, 0.4, 100), 2 * np.pi)
    out = fisher_lee_circular(a, b, n_perm=200, seed=0)
    assert out["r"] > 0.5
    assert out["p"] < 0.05


def test_fisher_lee_independent():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 2 * np.pi, 100)
    b = rng.uniform(0, 2 * np.pi, 100)
    out = fisher_lee_circular(a, b, n_perm=200, seed=0)
    assert out["p"] > 0.05


def test_js_rank_coupled():
    rng = np.random.default_rng(4)
    a = rng.vonmises(1.0, 0.8, 120)
    b = np.mod(-a + rng.normal(0, 0.4, 120), 2 * np.pi)
    out = jammalamadaka_sarma(a, b, n_perm=200, seed=0)
    assert out["r"] < -0.3


def test_input_validation():
    with pytest.raises(ValueError):
        mardia_circular_linear(np.array([0.1, 0.2]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fisher_lee_circular(np.array([0.1]), np.array([0.2]))
    with pytest.raises(ValueError):
        jammalamadaka_sarma(np.array([0.1, np.nan]), np.array([0.2, 0.3]))


def test_bench_passes():
    out = bench_circular_correlation()
    assert out["synthetic_mardia_dep_p"] < 0.05
    assert out["synthetic_mardia_ind_p"] > 0.05
    assert out["synthetic_fisher_lee_r"] > 0.5
    assert out["synthetic_js_r"] > 0.3
    assert out["synthetic_score"] == 1.0
