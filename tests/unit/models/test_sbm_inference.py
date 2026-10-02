"""DC-SBM inference tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sbm_inference import (
    bench_sbm,
    dcsbm_loglik,
    nmi,
    spectral_sbm,
)


def _planted(n=180, k=3, seed=0, p_in=0.25, p_out=0.02):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    B = np.full((k, k), p_out)
    np.fill_diagonal(B, p_in)
    U = rng.random((n, n))
    A = (B[z][:, z] > U) & np.triu(np.ones((n, n))).astype(bool)
    return (A + A.T).astype(float), z


def test_spectral_recovers_partition():
    A, z = _planted()
    z_hat = spectral_sbm(A, 3, 0)
    assert nmi(z, z_hat) > 0.85


def test_nmi_properties():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert nmi(a, a) == pytest.approx(1.0)
    b = np.array([0, 1, 0, 1, 0, 1])
    assert 0 <= nmi(a, b) < 0.5


def test_dcsbm_loglik_better_on_true():
    A, z = _planted()
    rng = np.random.default_rng(9)
    z_rand = rng.integers(0, 3, z.size)
    assert dcsbm_loglik(A, z) > dcsbm_loglik(A, z_rand)


def test_bench_sbm():
    out = bench_sbm()
    assert out["synthetic_nmi"] > 0.9
