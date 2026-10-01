"""Tests for wasserstein: W1/W2 OT metrics, Sinkhorn, energy distance bridge.

Every random draw uses a seeded np.random.default_rng, so runs are
deterministic. Numerical anchors: the 1-D W1 of N(0,1) vs N(1,1) samples,
the closed-form Gaussian W2 (Gelbrich 1990; Bhatia-Jain-Lim 2019), the
Sinkhorn-to-assignment limit as reg -> 0 (Cuturi 2013), the debiased
Sinkhorn divergence (Genevay-Peyre-Cuturi 2018), and fail-closed edges.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from quant_fund.metrics.wasserstein import (
    energy_distance_from_ot,
    sinkhorn_distance,
    sinkhorn_divergence,
    wasserstein_1d,
    wasserstein_gaussian,
)

Array = NDArray[np.float64]

SEED = 20260928


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------- W1, 1-D


def test_w1_point_mass_is_distance() -> None:
    a = np.array([0.0])
    b = np.array([3.0])
    assert wasserstein_1d(a, b) == pytest.approx(3.0)
    assert wasserstein_1d(a, b, p=2) == pytest.approx(3.0)
    assert wasserstein_1d(a, a) == pytest.approx(0.0)


def test_w1_gaussian_shift_within_mc_tolerance() -> None:
    rng = _rng()
    x = rng.normal(0.0, 1.0, size=8000)
    y = rng.normal(1.0, 1.0, size=8000)
    # Exact W1 = 1 for N(0,1) vs N(1,1); the rank coupling is exact in 1-D,
    # so only quantile Monte-Carlo error remains (O(1/sqrt(n)) tails).
    assert wasserstein_1d(x, y) == pytest.approx(1.0, abs=0.05)


def test_w1_identical_samples_zero() -> None:
    rng = _rng()
    x = rng.normal(size=1000)
    assert wasserstein_1d(x, x) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- Gaussian closed form


def test_w2_gaussian_identical_is_zero() -> None:
    mu = np.array([0.5, -1.0, 2.0])
    cov = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 1.5]])
    assert wasserstein_gaussian(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-10)


def test_w2_gaussian_shift_is_norm_of_delta() -> None:
    d = 4
    cov = np.eye(d)
    mu1 = np.zeros(d)
    mu2 = np.array([3.0, 0.0, 4.0, 0.0])  # ||delta|| = 5
    assert wasserstein_gaussian(mu1, cov, mu2, cov) == pytest.approx(25.0, abs=1e-10)


def test_w2_gaussian_scaled_isotropic_trace_term() -> None:
    # S1 = I, S2 = 4I: S1^{1/2} S2 S1^{1/2} = 4I, sqrt = 2I, so
    # tr(S1 + S2 - 2*2I) = tr(I + 4I - 4I) = tr(I) = d = (1-2)^2 * d.
    d = 3
    cov1 = np.eye(d)
    cov2 = 4.0 * np.eye(d)
    mu = np.zeros(d)
    assert wasserstein_gaussian(mu, cov1, mu, cov2) == pytest.approx(float(d), abs=1e-10)
    # And with means, mean term adds in quadrature.
    mu2 = np.ones(d)
    assert wasserstein_gaussian(mu, cov1, mu2, cov2) == pytest.approx(
        float(d) + float(d), abs=1e-10
    )


def test_w2_gaussian_correlated_analytic_value() -> None:
    # Diagonal commuting case: S1 = diag(1, 4), S2 = diag(4, 9).
    # sqrt(S1 S2) in the commuting case is sqrt(S1) @ sqrt(S2) =
    # diag(1*2, 2*3) = diag(2, 6); tr term = (1+4-2*2) + (4+9-2*6) = 1 + 1 = 2.
    cov1 = np.diag([1.0, 4.0])
    cov2 = np.diag([4.0, 9.0])
    mu = np.zeros(2)
    assert wasserstein_gaussian(mu, cov1, mu, cov2) == pytest.approx(2.0, abs=1e-10)


# -------------------------------------------- Sinkhorn vs exact OT


def test_sinkhorn_converges_to_assignment_as_reg_small() -> None:
    rng = _rng()
    a = rng.normal(size=(5, 2))
    b = rng.normal(size=(5, 2)) + 1.0
    cost = cdist(a, b, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    exact = float(cost[rows, cols].sum()) / 5.0  # equal uniform weights
    approx = sinkhorn_distance(a, b, reg=0.01, tol=1e-12, max_iter=5000)
    assert approx == pytest.approx(exact, abs=0.02)


def test_sinkhorn_uniform_weights_match_default() -> None:
    rng = _rng()
    a = rng.normal(size=(6, 3))
    b = rng.normal(size=(4, 3)) + 0.5
    explicit = sinkhorn_distance(
        a, b, weights_a=np.full(6, 1.0 / 6.0), weights_b=np.full(4, 1.0 / 4.0)
    )
    assert explicit == pytest.approx(sinkhorn_distance(a, b), abs=1e-12)


# ------------------------------------------------- Sinkhorn divergence


def test_sinkhorn_divergence_identical_is_zero() -> None:
    rng = _rng()
    a = rng.normal(size=(30, 2))
    # A == B: S(A,B), S(A,A), S(B,B) are the same computation, so the
    # divergence cancels exactly (bit-for-bit identical summands).
    assert sinkhorn_divergence(a, a) == 0.0


def test_sinkhorn_divergence_positive_for_shifted() -> None:
    rng = _rng()
    a = rng.normal(size=(200, 2))
    b = rng.normal(size=(200, 2)) + 2.0
    same_law_other_draw = rng.normal(size=(200, 2))
    d_shift = sinkhorn_divergence(a, b, reg=0.1)
    d_same = sinkhorn_divergence(a, same_law_other_draw, reg=0.1)
    assert d_shift > 0.0
    assert d_shift > d_same  # a 2-sigma mean shift dwarfs same-law noise


def test_sinkhorn_divergence_is_symmetric() -> None:
    rng = _rng()
    a = rng.normal(size=(40, 3))
    b = rng.normal(size=(25, 3)) + 1.0
    assert sinkhorn_divergence(a, b) == pytest.approx(sinkhorn_divergence(b, a), abs=1e-9)


# ------------------------------------------- Triangle + symmetry


def _triples() -> tuple[Array, Array, Array]:
    rng = _rng()
    return (
        rng.normal(size=(50, 2)),
        rng.normal(size=(50, 2)) + 1.0,
        rng.normal(size=(50, 2)) * 1.5,
    )


def test_w1_triangle_inequality() -> None:
    a, b, c = _triples()
    da_b = wasserstein_1d(a[:, 0], b[:, 0])
    d_b_c = wasserstein_1d(b[:, 0], c[:, 0])
    d_a_c = wasserstein_1d(a[:, 0], c[:, 0])
    assert d_a_c <= da_b + d_b_c + 1e-12


def test_w1_symmetry() -> None:
    a, b, _ = _triples()
    assert wasserstein_1d(a[:, 0], b[:, 0]) == pytest.approx(
        wasserstein_1d(b[:, 0], a[:, 0]), abs=1e-12
    )


def test_sinkhorn_divergence_triangle_inequality() -> None:
    a, b, c = _triples()
    d_ab = sinkhorn_divergence(a, b, reg=0.5)
    d_bc = sinkhorn_divergence(b, c, reg=0.5)
    d_ac = sinkhorn_divergence(a, c, reg=0.5)
    assert d_ac <= d_ab + d_bc + 1e-9


# ------------------------------------------- Energy distance bridge


def test_energy_distance_identical_samples_nonnegative_small() -> None:
    rng = _rng()
    a = rng.normal(size=(300, 2))
    d2 = energy_distance_from_ot(a, a)
    assert d2 == pytest.approx(0.0, abs=1e-12)


def test_energy_distance_positive_for_shift_and_bounds_w1() -> None:
    rng = _rng()
    a = rng.normal(size=(400, 2))
    b = rng.normal(size=(400, 2)) + 1.0
    d2 = energy_distance_from_ot(a, b)
    assert d2 > 0.0
    # D^2 <= 2 W1 for first-moment costs; use a deterministic 1-D W1 anchor.
    w1 = wasserstein_1d(a[:, 0], b[:, 0]) + wasserstein_1d(a[:, 1], b[:, 1])
    assert d2 <= 2.0 * w1 + 0.1  # loose coupling bound under MC noise


def test_energy_distance_matches_energy_score_point_mass_view() -> None:
    # D^2(ensemble, point mass at y) = 2 E||X - y|| - E||X - X'|| - 0, so
    # the bridge equals 2 * E||X - y|| - E||X - X'||; the second term is the
    # unbiased within-ensemble mean (off-diagonal), matching energy_score's
    # second term times 2n/(n-1).
    from quant_fund.metrics.energy_score import energy_score

    rng = _rng()
    x = rng.normal(size=(200, 3))
    y = np.array([0.5, -0.5, 1.0])
    d2 = energy_distance_from_ot(x, y.reshape(1, -1))
    es = energy_score(x, y)
    n = x.shape[0]
    within = float(np.sum(cdist(x, x))) / float(n * n)  # biased (with diagonal)
    expected = 2.0 * float(np.mean(cdist(x, y.reshape(1, -1)))) - within
    assert d2 == pytest.approx(expected, abs=1e-12)
    # Cross-check against the proper score up to the diagonal correction.
    assert d2 == pytest.approx(2.0 * es + within / (n - 1.0), rel=1e-6)


def test_energy_distance_symmetric() -> None:
    rng = _rng()
    a = rng.normal(size=(100, 2))
    b = rng.normal(size=(80, 2)) + 0.3
    assert energy_distance_from_ot(a, b) == pytest.approx(energy_distance_from_ot(b, a), abs=1e-12)


# ------------------------------------------------- Fail-closed edges


def test_wasserstein_1d_fail_closed() -> None:
    with pytest.raises(ValueError):
        wasserstein_1d(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        wasserstein_1d(np.array([np.nan, 1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        wasserstein_1d(np.array([1.0]), np.array([2.0]), p=0.5)
    with pytest.raises(ValueError):
        wasserstein_1d(np.array([1.0]), np.array([2.0]), p=np.nan)


def test_wasserstein_gaussian_fail_closed() -> None:
    mu = np.zeros(2)
    cov = np.eye(2)
    with pytest.raises(ValueError):
        wasserstein_gaussian(np.zeros(2), cov, np.zeros(3), cov)  # dim mismatch
    with pytest.raises(ValueError):
        wasserstein_gaussian(mu, np.ones((2, 3)), mu, cov)  # non-square
    with pytest.raises(ValueError):
        wasserstein_gaussian(mu, np.diag([1.0, -1.0]), mu, cov)  # non-PSD
    with pytest.raises(ValueError):
        wasserstein_gaussian(mu, np.array([[np.nan, 0.0], [0.0, 1.0]]), mu, cov)
    with pytest.raises(ValueError):
        wasserstein_gaussian(mu, np.array([[1.0, 2.0], [0.0, 1.0]]), mu, cov)  # asymmetric


def test_sinkhorn_fail_closed() -> None:
    rng = _rng()
    a = rng.normal(size=(5, 2))
    b = rng.normal(size=(6, 3))
    with pytest.raises(ValueError):
        sinkhorn_distance(a, b)  # dim mismatch
    with pytest.raises(ValueError):
        sinkhorn_distance(a[:0], a)  # empty
    with pytest.raises(ValueError):
        sinkhorn_distance(np.array([[np.inf, 0.0]]), a)  # non-finite
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, reg=0.0)
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, reg=-1.0)
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, weights_a=np.full(4, 0.25))  # wrong length
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, weights_a=np.array([1.0, -1.0, 1.0, 1.0, 1.0]))  # negative
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, weights_a=np.zeros(5))  # zero total mass
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, max_iter=0)
    with pytest.raises(ValueError):
        sinkhorn_distance(a, a, tol=0.0)
    with pytest.raises(ValueError):
        sinkhorn_divergence(a, b)  # dim mismatch in divergence too
