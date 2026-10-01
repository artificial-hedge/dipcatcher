"""Graded-response / partial-credit IRT tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.graded_irt import (
    bench_graded_irt,
    grm_jml,
    pcm_jml,
)


def _simulate_grm(seed: int, n_p: int = 60, n_i: int = 8, k_cat: int = 4):
    from quant_fund.models.graded_irt import _grm_probs

    rng = np.random.default_rng(seed)
    theta_true = rng.normal(0, 1, n_p)
    a_true = rng.uniform(0.8, 1.6, n_i)
    b_true = np.sort(rng.normal(0, 0.9, (n_i, k_cat - 1)), axis=1)
    resp = np.zeros((n_p, n_i), dtype=np.int64)
    for i in range(n_p):
        for j in range(n_i):
            p = _grm_probs(theta_true[i], a_true[j], np.sort(b_true[j]))
            resp[i, j] = rng.choice(k_cat, p=p / p.sum())
    return resp, theta_true, a_true


def test_grm_recovers_theta_ordering():
    resp, theta_true, _ = _simulate_grm(0)
    fit = grm_jml(resp, n_iter=10)
    rho = np.corrcoef(theta_true, np.asarray(fit["theta"]))[0, 1]
    assert rho > 0.6


def test_grm_shapes():
    resp, _, _ = _simulate_grm(1, n_p=30, n_i=5)
    fit = grm_jml(resp, n_iter=5)
    assert np.asarray(fit["theta"]).shape == (30,)
    assert np.asarray(fit["a"]).shape == (5,)
    assert np.asarray(fit["b"]).shape == (5, 3)
    assert np.isfinite(fit["loglik"])


def test_pcm_recovers_theta_ordering():
    rng = np.random.default_rng(2)
    n_p, n_i, k_cat = 50, 6, 4
    theta_true = rng.normal(0, 1, n_p)
    d_true = np.sort(rng.normal(0, 0.7, (n_i, k_cat - 1)), axis=1)
    resp = np.zeros((n_p, n_i), dtype=np.int64)
    from quant_fund.models.graded_irt import _pcm_probs

    for i in range(n_p):
        for j in range(n_i):
            p = _pcm_probs(theta_true[i], d_true[j])
            resp[i, j] = rng.choice(k_cat, p=p / p.sum())
    fit = pcm_jml(resp, n_iter=10)
    rho = np.corrcoef(theta_true, np.asarray(fit["theta"]))[0, 1]
    assert rho > 0.6


def test_pcm_theta_standardized():
    resp, _, _ = _simulate_grm(3, n_p=40, n_i=5)
    fit = pcm_jml(resp, n_iter=5)
    th = np.asarray(fit["theta"])
    assert abs(th.mean()) < 1e-6
    assert abs(th.std() - 1.0) < 0.2


def test_input_validation():
    with pytest.raises(ValueError):
        grm_jml(np.array([[0, 1, -1], [1, 0, 2]]))
    with pytest.raises(ValueError):
        pcm_jml(np.array([[0, 1, -1], [1, 0, 2]], dtype=np.int64))


def test_bench_passes():
    out = bench_graded_irt()
    assert out["synthetic_grm_theta_rho"] > 0.5
    assert out["synthetic_pcm_theta_rho"] > 0.6
    assert out["synthetic_score"] == 1.0
