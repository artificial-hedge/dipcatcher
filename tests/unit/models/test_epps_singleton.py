"""Tests for epps_singleton — ECF normality test."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.epps_singleton import bench_epps_singleton, epps_singleton


def test_accepts_normal():
    rng = np.random.default_rng(0)
    out = epps_singleton(rng.standard_normal(500))
    assert out["p"] > 0.05
    assert out["g1"] >= 0


def test_rejects_lognormal():
    rng = np.random.default_rng(1)
    assert epps_singleton(rng.lognormal(size=500))["p"] < 0.05


def test_rejects_heavy_tail():
    rng = np.random.default_rng(2)
    assert epps_singleton(rng.standard_t(3.0, size=500))["p"] < 0.05


def test_bad_inputs():
    with pytest.raises(ValueError):
        epps_singleton(np.ones(5))
    with pytest.raises(ValueError):
        epps_singleton(np.ones(50))


def test_bench():
    assert bench_epps_singleton()["score"] == 1.0
