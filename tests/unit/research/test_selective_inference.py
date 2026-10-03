"""Tests for research/selective_inference.py — Lee–Sun conditional CIs."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.selective_inference import (
    SELECTIVE_SCHEMA,
    _tn_cdf,
    argmin_polytope,
    selective_bench,
    selective_ci,
    truncation_interval,
)


def test_argmin_polytope_encodes_winner() -> None:
    A, b = argmin_polytope(1, 4)
    assert A.shape == (3, 4)
    z = np.array([0.5, 0.1, 0.2, 0.3])  # argmin = 1
    assert np.all(A @ z <= b)
    z_bad = np.array([0.05, 0.1, 0.2, 0.3])  # argmin = 0, violates
    assert not np.all(A @ z_bad <= b)


def test_tn_cdf_standard_cases() -> None:
    # untruncated within range, symmetric: x=mu → 0.5 when lo=-inf, hi=+inf
    assert _tn_cdf(0.0, 0.0, 1.0, -np.inf, np.inf) == pytest.approx(0.5)
    # right edge → 1; left edge → 0
    assert _tn_cdf(5.0, 0.0, 1.0, -np.inf, 5.0) == pytest.approx(1.0)
    assert _tn_cdf(-3.0, 0.0, 1.0, -3.0, np.inf) == pytest.approx(0.0)
    # deep-tail stability: mu far above a finite hi returns ~0 for x < hi
    assert _tn_cdf(0.0, 500.0, 1.0, -np.inf, 0.5) < 1e-6
    with pytest.raises(ValueError):
        _tn_cdf(0.0, 0.0, 0.0, -1.0, 1.0)


def test_truncation_interval_argmin() -> None:
    k = 4
    z = np.array([0.3, 0.1, 0.4, 0.2])
    Sigma = np.eye(k) * 0.01
    A, b = argmin_polytope(1, k)
    eta = np.zeros(k)
    eta[1] = 1.0
    lo, hi = truncation_interval(z, A, b, eta, Sigma)
    assert lo == -np.inf
    # v_plus: z_1 can rise until it hits the smallest rival mean
    assert hi == pytest.approx(0.2, abs=1e-9)


def test_selective_ci_brackets_truth() -> None:
    rng = np.random.default_rng(3)
    k = 3
    true_means = np.array([0.05, 0.1, 0.15])
    Sigma = np.eye(k) * (1.0 / 200)
    covered = 0
    n = 150
    for _ in range(n):
        means = rng.multivariate_normal(true_means, Sigma)
        ci = selective_ci(means, Sigma, alpha=0.1)
        if ci.feasible and ci.lo <= true_means[ci.winner] <= ci.hi:
            covered += 1
    assert covered / n > 0.8  # >= nominal 0.9 within MC tolerance


def test_selective_ci_wider_than_naive() -> None:
    means = np.array([0.05, 0.05, 0.5])
    Sigma = np.eye(3) * 0.01
    ci = selective_ci(means, Sigma)
    assert ci.feasible
    assert (ci.hi - ci.lo) >= (ci.naive_hi - ci.naive_lo)


def test_bench_smoke_and_schema() -> None:
    out = selective_bench(n_reps=60, k=4)
    assert out["schema"] == SELECTIVE_SCHEMA
    assert 0.0 <= out["coverage_conditional"] <= 1.0
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
