"""Tests for cov_int — covariance intersection fusion."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cov_int import bench_cov_int, cov_int


def test_correlated_error_model() -> None:
    # bench claims "unknown cross-correlation in truth" — the draw must
    # really carry covariances A, B AND a nonzero rho-scaled cross-covariance
    # (the old code overwrote the draw: b ended independent and a overdispersed).
    from quant_fund.models.cov_int import _draw_ab

    rng = np.random.default_rng(0)
    tru = np.zeros(2)
    A = np.array([[1.0, 0.2], [0.2, 0.5]])
    B = np.array([[0.8, -0.1], [-0.1, 0.4]])
    rho = 0.5
    La = np.linalg.cholesky(A)
    Lb = np.linalg.cholesky(B)
    draws = np.array([_draw_ab(rng, tru, La, Lb, rho) for _ in range(6000)])
    ea = draws[:, 0, :]
    eb = draws[:, 1, :]
    cov_a = np.cov(ea.T)
    cov_b = np.cov(eb.T)
    cross = (ea.T @ eb) / len(ea)
    assert np.allclose(cov_a, A, atol=0.08)
    assert np.allclose(cov_b, B, atol=0.08)
    assert np.allclose(cross, rho * La @ Lb.T, atol=0.08)


def test_identical_inputs_fuse_to_self() -> None:
    A = np.array([[1.0, 0.2], [0.2, 0.5]])
    a = np.array([1.0, -2.0])
    c, C, _w = cov_int(a, A, a, A)
    assert np.allclose(c, a)
    assert np.trace(C) <= np.trace(A) + 1e-12


def test_bench_cov_int() -> None:
    out = bench_cov_int()
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
    assert out["synthetic_ci_consistent"] == 1.0
