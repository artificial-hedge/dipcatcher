"""Tests for latent_class — Bernoulli LCA EM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.latent_class import bench_latent_class, latent_class_em


def _mixture(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    p_t = np.array([[0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.8, 0.8]])
    z = (rng.random(n) < 0.4).astype(int)
    x = (rng.random((n, 4)) < p_t[z]).astype(float)
    return x, z


def test_em_recovers_classes():
    x, z = _mixture()
    fit = latent_class_em(x, n_classes=2, seed=0)
    assign = np.asarray(fit["assign"])
    acc = max(float((assign == z).mean()), float((assign != z).mean()))
    assert acc > 0.85


def test_resp_rows_sum_to_one():
    x, _ = _mixture()
    fit = latent_class_em(x, n_classes=2, seed=0)
    resp = np.asarray(fit["resp"])
    assert np.allclose(resp.sum(axis=1), 1.0)


def test_bic_and_loglik_finite():
    x, _ = _mixture()
    fit = latent_class_em(x, n_classes=2, seed=0)
    assert np.isfinite(fit["loglik"])
    assert np.isfinite(fit["bic"])


def test_fail_closed_nonbinary():
    with pytest.raises(ValueError):
        latent_class_em(np.array([[0.5] * 4] * 8))


def test_bench():
    out = bench_latent_class()
    assert out["score"] == 1.0
