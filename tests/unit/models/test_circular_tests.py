"""Circular two-sample and uniformity tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import vonmises

from quant_fund.models.circular_tests import (
    bench_circular_tests,
    circular_runs,
    mardia_watson_wheeler,
    rao_spacing,
)


def test_mww_separated_groups():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, 100)
    y = rng.uniform(np.pi, 2 * np.pi, 100)
    out = mardia_watson_wheeler(x, y, n_perm=400, seed=0)
    assert out["p"] < 0.05
    assert out["stat"] > 0


def test_mww_same_distribution():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 2 * np.pi, 120)
    y = rng.uniform(0, 2 * np.pi, 120)
    out = mardia_watson_wheeler(x, y, n_perm=400, seed=0)
    assert out["p"] > 0.05


def test_rao_clustered_vs_uniform():
    rng = np.random.default_rng(2)
    unif = rng.uniform(0, 2 * np.pi, 150)
    clus = vonmises.rvs(2.0, loc=0.0, size=150, random_state=rng)
    assert rao_spacing(clus)["p"] < rao_spacing(unif)["p"]


def test_rao_rejects_clustering():
    rng = np.random.default_rng(3)
    clus = vonmises.rvs(3.0, loc=1.0, size=200, random_state=rng)
    assert rao_spacing(clus)["p"] < 0.01


def test_circular_runs_detects_blocks():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, np.pi, 80)
    y = rng.uniform(np.pi, 2 * np.pi, 80)
    out = circular_runs(x, y)
    assert out["p"] < 0.01


def test_input_validation():
    with pytest.raises(ValueError):
        mardia_watson_wheeler(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        rao_spacing(np.array([0.1, np.nan, 0.3]))


def test_bench_passes():
    out = bench_circular_tests()
    for key in (
        "synthetic_mww_same_p",
        "synthetic_mww_diff_p",
        "synthetic_rao_unif_p",
        "synthetic_rao_clus_p",
    ):
        assert 0.0 <= out[key] <= 1.0
    assert out["synthetic_mww_same_p"] > out["synthetic_mww_diff_p"]
    assert out["synthetic_rao_unif_p"] > out["synthetic_rao_clus_p"]
    assert out["synthetic_score"] == 1.0
