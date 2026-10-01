"""Tests for energy_test — Szekely-Rizzo energy MVN test."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.energy_test import bench_energy_test, energy_mvn_test


def test_accepts_mvn():
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(np.zeros(3), np.eye(3), size=150)
    out = energy_mvn_test(x, n_boot=100, seed=0)
    assert out["p"] > 0.01
    assert np.isfinite(out["e_stat"])


def test_rejects_heavy_tail():
    rng = np.random.default_rng(1)
    x = rng.standard_t(2.0, size=(150, 3)) * 3.0
    out = energy_mvn_test(x, n_boot=100, seed=1)
    assert out["p"] < 0.1


def test_deterministic_with_seed():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((80, 2))
    a = energy_mvn_test(x, n_boot=60, seed=7)
    b = energy_mvn_test(x, n_boot=60, seed=7)
    assert a == b


def test_bad_inputs():
    with pytest.raises(ValueError):
        energy_mvn_test(np.ones((4, 2)))
    with pytest.raises(ValueError):
        energy_mvn_test(np.ones((10, 8)))


def test_bench():
    assert bench_energy_test()["score"] == 1.0
