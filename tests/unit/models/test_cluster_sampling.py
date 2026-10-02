"""Tests for cluster_sampling — two-stage design."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cluster_sampling import (
    bench_cluster_sampling,
    cluster_mean,
    cluster_total,
)


def test_cluster_total_se():
    rng = np.random.default_rng(0)
    out = cluster_total(rng.gamma(2.0, 10.0, 30), np.full(30, 2.0))
    assert out["se"] > 0
    assert out["total"] > 0


def test_icc_detected():
    rng = np.random.default_rng(1)
    n_cl, per = 40, 10
    mu = rng.normal(0, 2.0, n_cl)
    y = np.concatenate([mu[i] + 0.5 * rng.standard_normal(per) for i in range(n_cl)])
    g = np.repeat(np.arange(n_cl), per).astype(float)
    out = cluster_mean(y, g)
    assert out["icc"] > 0.3
    assert out["deff_cluster"] > 1.5


def test_iid_low_icc():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(200)
    g = np.repeat(np.arange(40), 5).astype(float)
    out = cluster_mean(y, g)
    assert out["icc"] < 0.3


def test_fpc_applied():
    rng = np.random.default_rng(3)
    a = cluster_total(rng.normal(50, 5, 20), np.full(20, 3.0), n_psu_population=100)
    b = cluster_total(rng.normal(50, 5, 20), np.full(20, 3.0))
    assert np.isfinite(a["se"]) and np.isfinite(b["se"])


def test_fail_closed_one_cluster():
    with pytest.raises(ValueError):
        cluster_mean(np.arange(10.0), np.zeros(10))


def test_bench():
    out = bench_cluster_sampling()
    assert out["score"] == 1.0
