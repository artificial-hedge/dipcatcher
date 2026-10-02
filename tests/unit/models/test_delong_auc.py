"""DeLong AUC module tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.delong_auc import (
    auc_wilcoxon,
    bench_delong_auc,
    delong_auc,
    delong_compare,
    delong_multi,
)


def test_auc_wilcoxon_perfect_and_random():
    rng = np.random.default_rng(0)
    pos = rng.normal(3.0, 0.5, 50)
    neg = rng.normal(0.0, 0.5, 50)
    assert auc_wilcoxon(pos, neg) > 0.99
    pos2 = rng.normal(0.0, 1.0, 100)
    neg2 = rng.normal(0.0, 1.0, 100)
    assert abs(auc_wilcoxon(pos2, neg2) - 0.5) < 0.15


def test_delong_auc_separable():
    rng = np.random.default_rng(1)
    fit = delong_auc(rng.normal(1.5, 1.0, 60), rng.normal(0.0, 1.0, 70))
    assert fit["auc"] > 0.75
    assert fit["p_half"] < 0.01
    assert fit["se"] > 0


def test_delong_auc_null():
    rng = np.random.default_rng(2)
    fit = delong_auc(rng.normal(0, 1, 80), rng.normal(0, 1, 80))
    assert fit["p_half"] > 0.05


def test_delong_compare_paired():
    rng = np.random.default_rng(3)
    lat_p = rng.normal(1.0, 1.0, 60)
    lat_n = rng.normal(0.0, 1.0, 60)
    strong_p = lat_p + rng.normal(0, 0.2, 60)
    strong_n = lat_n + rng.normal(0, 0.2, 60)
    weak_p = lat_p + rng.normal(0, 1.5, 60)
    weak_n = lat_n + rng.normal(0, 1.5, 60)
    out = delong_compare(weak_p, weak_n, strong_p, strong_n)
    assert out["diff"] < 0
    assert out["se_diff"] > 0


def test_delong_multi_covariance():
    rng = np.random.default_rng(4)
    k = 3
    sp = rng.normal(1.0, 1.0, (50, k))
    sn = rng.normal(0.0, 1.0, (50, k))
    out = delong_multi(sp, sn)
    cov = np.asarray(out["cov"])
    assert cov.shape == (k, k)
    np.testing.assert_allclose(cov, cov.T)
    assert (np.diag(cov) > 0).all()


def test_input_validation():
    with pytest.raises(ValueError):
        delong_auc(np.array([1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        delong_compare(
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            np.array([0.0, 1.0]),
        )


def test_bench_passes():
    out = bench_delong_auc()
    assert out["synthetic_auc"] > 0.7
    assert out["synthetic_p_sep"] < 0.01
    assert out["synthetic_p_null"] > 0.05
    assert out["synthetic_score"] == 1.0
