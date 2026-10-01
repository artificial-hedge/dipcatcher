"""von Mises-Fisher mixture EM tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.vonmises_fisher import (
    bench_vonmises_fisher,
    movm_em,
    vmf_fit,
)


def _tight_cluster(mu: np.ndarray, kappa: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, mu.shape[0]))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    # blend toward mu with weight ~ kappa/(kappa+...) then renormalize
    w = 0.85 if kappa > 5 else 0.5
    x = w * mu[None, :] + (1.0 - w) * x
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_vmf_fit_recovers_direction():
    mu = np.array([0.0, 0.0, 1.0])
    x = _tight_cluster(mu, 10.0, 200, 0)
    fit = vmf_fit(x)
    cos = float(np.asarray(fit["mu"]) @ mu)
    assert cos > 0.95
    assert float(fit["kappa"]) > 1.0


def test_vmf_fit_concentration():
    mu = np.array([1.0, 0.0, 0.0])
    x = _tight_cluster(mu, 20.0, 400, 1)
    fit = vmf_fit(x)
    assert float(fit["r_bar"]) > 0.8
    assert np.isfinite(fit["loglik"])


def test_movm_two_clusters():
    mu1 = np.array([1.0, 0.0, 0.0])
    mu2 = np.array([0.0, 1.0, 0.0])
    x = np.vstack([_tight_cluster(mu1, 15.0, 150, 0), _tight_cluster(mu2, 15.0, 150, 1)])
    fit = movm_em(x, 2, seed=0, n_iter=60)
    mus = np.asarray(fit["mus"])
    cos = np.abs(mus @ mu1)
    assert cos.max() > 0.95
    assert np.abs(mus @ mu2).max() > 0.95


def test_movm_single_component():
    x = _tight_cluster(np.array([0.0, 0.0, 1.0]), 8.0, 100, 3)
    fit = movm_em(x, 1, seed=0)
    assert np.isfinite(fit["loglik"])
    assert float(np.asarray(fit["kappas"])[0]) > 0


def test_input_validation():
    with pytest.raises(ValueError):
        vmf_fit(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        movm_em(np.random.default_rng(0).normal(size=(10, 3)), 0)
    with pytest.raises(ValueError):
        vmf_fit(np.ones((3, 2)) / np.sqrt(2))  # n<4


def test_bench_passes():
    out = bench_vonmises_fisher(seed=11)
    assert out["synthetic_cos_min"] > 0.95
    assert out["synthetic_purity"] > 0.9
    assert out["synthetic_kappa_max_err"] < 6.0
    assert out["synthetic_score"] == 1.0
