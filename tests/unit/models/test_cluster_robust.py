"""Tests for cluster-robust inference (models/cluster_robust.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.cluster_robust import (
    bench_cluster_robust,
    crve,
    synth_cluster,
    wild_cluster_bootstrap,
)


def test_crve_wider_than_iid_under_clustering():
    d = synth_cluster(n_clusters=30, seed=5, rho=0.7)
    y, x, cl = np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"])
    cv = crve(y, x, cl)
    n = y.size
    a = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    xtx = np.linalg.pinv(a.T @ a)
    s2 = float(resid @ resid / (n - 2))
    se_iid = math.sqrt(s2 * xtx[1, 1])
    assert cv["se"][1] > se_iid


def test_crve_null_not_rejected():
    d = synth_cluster(seed=6, beta=0.0, rho=0.5)
    cv = crve(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"]))
    assert cv["p"][1] > 0.01


def test_crve_power():
    d = synth_cluster(seed=7, beta=0.5, rho=0.5)
    cv = crve(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"]))
    assert cv["p"][1] < 0.05
    assert cv["beta"][1] > 0.2


def test_cr1_larger_than_cr0():
    d = synth_cluster(seed=8, beta=0.2)
    y, x, cl = np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"])
    assert crve(y, x, cl, corr="cr1")["se"][1] >= crve(y, x, cl, corr="cr0")["se"][1]


def test_wild_bootstrap_null_and_power():
    d0 = synth_cluster(seed=9, beta=0.0, rho=0.5)
    wb0 = wild_cluster_bootstrap(
        np.asarray(d0["y"]), np.asarray(d0["x"]), np.asarray(d0["cluster"]), n_boot=399, seed=9
    )
    assert wb0["p_wild"] > 0.01
    d1 = synth_cluster(seed=10, beta=0.6, rho=0.4)
    wb1 = wild_cluster_bootstrap(
        np.asarray(d1["y"]), np.asarray(d1["x"]), np.asarray(d1["cluster"]), n_boot=399, seed=10
    )
    assert wb1["p_wild"] < 0.05


def test_webb_weights_few_clusters():
    d = synth_cluster(n_clusters=6, n_per=30, seed=11, rho=0.6)
    wb = wild_cluster_bootstrap(
        np.asarray(d["y"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
        n_boot=299,
        weight="webb",
        seed=11,
    )
    assert 0.0 < wb["p_wild"] <= 1.0


def test_validation():
    with pytest.raises(ValueError):
        crve(np.ones(5), np.ones(5), np.array([0, 0, 1, 1, 2]))
    with pytest.raises(ValueError):
        crve(
            np.random.default_rng(0).normal(size=20),
            np.random.default_rng(0).normal(size=20),
            np.zeros(20),
        )
    with pytest.raises(ValueError):
        wild_cluster_bootstrap(np.ones(20), np.ones(20), np.zeros(20), coef_idx=9)


def test_determinism():
    d = synth_cluster(seed=12, beta=0.3)
    a = wild_cluster_bootstrap(
        np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"]), n_boot=99, seed=12
    )
    b = wild_cluster_bootstrap(
        np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["cluster"]), n_boot=99, seed=12
    )
    assert a["p_wild"] == b["p_wild"]


def test_bench_keys():
    out = bench_cluster_robust()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_size_ok"] == 1.0
    assert out["synthetic_crve_power"] == 1.0
    assert out["synthetic_determinism"] == 1.0
