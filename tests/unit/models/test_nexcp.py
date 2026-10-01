"""Tests for quant_fund.models.nexcp — NexCP (Barber et al. 2023, arXiv:2202.13415).

SYNTHETIC data only: every coverage experiment below is a seeded synthetic
correctness test, never market evidence.
"""

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.nexcp import (
    NexCPSplit,
    kernel_weights,
    normalize_weights,
    recency_weights,
    weighted_conformal_quantile,
)

ALPHA = 0.1


def _calibrate(residuals: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> NexCPSplit:
    return NexCPSplit(alpha=alpha).calibrate(residuals, weights)


# ------------------------------------------------------------- weight helpers
def test_normalize_weights_matches_eq10() -> None:
    w_tilde, w_test = normalize_weights(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(w_tilde, np.full(3, 0.25))
    assert w_test == pytest.approx(0.25)
    assert float(w_tilde.sum()) + w_test == pytest.approx(1.0)
    # Unit weights recover the 1/(n+1) exchangeable scheme exactly (Eq. (10)
    # is deliberately not scale-invariant: the test atom always carries 1).
    w_tilde, w_test = normalize_weights(np.ones(199))
    np.testing.assert_allclose(w_tilde, np.full(199, 1.0 / 200.0))
    assert w_test == pytest.approx(1.0 / 200.0)
    scaled, scaled_test = normalize_weights(np.full(199, 0.37))
    assert scaled_test > w_test  # less total trust mass -> bigger test atom
    # Zero weights are allowed as long as total mass is positive.
    w_tilde, w_test = normalize_weights(np.array([0.0, 1.0]))
    np.testing.assert_allclose(w_tilde, np.array([0.0, 0.5]))
    assert w_test == pytest.approx(0.5)


def test_normalize_weights_fail_closed() -> None:
    with pytest.raises(ValueError):
        normalize_weights(np.array([]))
    with pytest.raises(ValueError):
        normalize_weights(np.array([-0.5, 1.0]))
    with pytest.raises(ValueError):
        normalize_weights(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        normalize_weights(np.zeros(5))  # degenerate: no trust mass at all


def test_recency_weights_increase_to_one() -> None:
    w = recency_weights(5, bandwidth=2.0)
    expected = np.exp(-np.array([4.0, 3.0, 2.0, 1.0, 0.0]) / 2.0)
    np.testing.assert_allclose(w, expected)
    assert w[-1] == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0.0)
    with pytest.raises(ValueError):
        recency_weights(0, bandwidth=1.0)
    with pytest.raises(ValueError):
        recency_weights(5, bandwidth=0.0)
    with pytest.raises(ValueError):
        recency_weights(5, bandwidth=np.inf)


def test_kernel_weights_decay_with_covariate_distance() -> None:
    x_cal = np.array([[0.0], [1.0], [2.0]])
    w = kernel_weights(x_cal, np.array([1.9]), bandwidth=1.0)
    np.testing.assert_allclose(w, np.exp(-np.array([1.9, 0.9, 0.1])))
    assert w[2] > w[1] > w[0]
    with pytest.raises(ValueError):
        kernel_weights(x_cal, np.array([1.9, 0.0]), bandwidth=1.0)  # dim mismatch
    with pytest.raises(ValueError):
        kernel_weights(np.empty((0, 1)), np.array([0.0]), bandwidth=1.0)
    with pytest.raises(ValueError):
        kernel_weights(x_cal, np.array([np.nan]), bandwidth=1.0)
    with pytest.raises(ValueError):
        kernel_weights(x_cal, np.array([0.0]), bandwidth=-1.0)


# -------------------------------------------------- weighted conformal quantile
def test_uniform_weights_recover_classical_split_conformal() -> None:
    rng = np.random.default_rng(11)
    r = np.abs(rng.normal(size=198)) + 0.01
    w = np.ones(198)
    q_nex = weighted_conformal_quantile(r, w, ALPHA)
    q_classic = conformal_quantile(r, ALPHA)
    assert q_nex == pytest.approx(q_classic, rel=1e-12)


def test_weighted_quantile_hand_computed_cases() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)  # w~ = 0.2 each, test atom 0.2
    # target mass 0.8 -> cum [0.2, 0.4, 0.6, 0.8] -> q = 4.0
    assert weighted_conformal_quantile(r, w, 0.2) == pytest.approx(4.0)
    # target mass 0.6 -> q = 3.0
    assert weighted_conformal_quantile(r, w, 0.4) == pytest.approx(3.0)
    # Heavier weight on the SMALL residuals pulls the quantile down.
    w2 = np.array([5.0, 1.0, 1.0, 1.0])  # w~ = [5/9, 1/9, 1/9, 1/9], w~_test = 1/9
    assert weighted_conformal_quantile(r, w2, 0.25) == pytest.approx(3.0)
    assert weighted_conformal_quantile(r, w2, 0.25) < weighted_conformal_quantile(r, w, 0.25)


def test_alpha_below_test_atom_raises_instead_of_infinity() -> None:
    r = np.abs(np.random.default_rng(3).normal(size=5)) + 0.1
    # w~_{n+1} = 1/6 > alpha = 0.1 -> the +inf atom is needed; fail closed.
    with pytest.raises(ValueError, match="no finite weighted conformal quantile"):
        weighted_conformal_quantile(r, np.ones(5), ALPHA)
    with pytest.raises(ValueError, match="effective sample size"):
        _calibrate(r, np.ones(5))


def test_weighted_quantile_fail_closed() -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([]), np.array([]), ALPHA)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0, np.nan]), np.ones(2), ALPHA)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(3), ALPHA)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(2), 0.0)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(2), 1.0)


# ------------------------------------------------------------- coverage bounds
def test_coverage_bounds_match_theorems_2_and_3() -> None:
    rng = np.random.default_rng(5)
    r = np.abs(rng.normal(size=50)) + 0.05
    w = recency_weights(50, bandwidth=10.0)
    m = _calibrate(r, w)
    w_tilde, w_test = normalize_weights(w)
    # Exchangeable case: gap = 0 -> [1 - alpha, 1 - alpha + w~_{n+1}].
    b0 = m.coverage_bounds(np.zeros(50))
    assert b0.gap == pytest.approx(0.0)
    assert b0.lower == pytest.approx(1.0 - ALPHA)
    assert b0.upper == pytest.approx(1.0 - ALPHA + w_test)
    # Worst case tv = 1: gap = sum_i w~_i = 1 - w~_{n+1}; None == ones.
    bw = m.coverage_bounds()
    b1 = m.coverage_bounds(np.ones(50))
    assert bw.gap == pytest.approx(b1.gap) == pytest.approx(1.0 - w_test)
    assert b1.lower == pytest.approx(max(0.0, w_test - ALPHA))
    assert b1.upper == pytest.approx(1.0)
    # Monotonicity: larger TV distances widen the gap.
    tv_mid = np.full(50, 0.5)
    assert m.coverage_bounds(tv_mid).gap == pytest.approx(float(w_tilde.sum()) * 0.5)
    assert b0.gap < m.coverage_bounds(tv_mid).gap < b1.gap
    # Bounds always bracket the nominal exchangeable coverage.
    assert b0.lower <= 1.0 - ALPHA <= b0.upper


def test_coverage_bounds_fail_closed() -> None:
    r = np.abs(np.random.default_rng(6).normal(size=20)) + 0.05
    m = _calibrate(r, np.ones(20))
    with pytest.raises(ValueError):
        m.coverage_bounds(np.full(19, 0.5))  # wrong length
    with pytest.raises(ValueError):
        m.coverage_bounds(np.full(20, 1.5))  # tv > 1
    with pytest.raises(ValueError):
        m.coverage_bounds(np.full(20, -0.1))  # tv < 0
    with pytest.raises(ValueError):
        m.coverage_bounds(np.r_[np.nan, np.zeros(19)])


# ------------------------------------------- SYNTHETIC shift coverage experiments
def test_coverage_under_covariate_shift_synthetic() -> None:
    """SYNTHETIC: heteroscedastic scale s(x) = 0.3 + 4x, test at x* = 0.9.

    Correctness test only (seeded synthetic draws), not market evidence.
    Uniform-weight split conformal averages stale small-x scales and
    undercovers at x*; the covariate-kernel NexCP weights concentrate on
    calibration points near x* and restore near-nominal coverage.
    """
    rng = np.random.default_rng(42)
    n, m_test = 400, 400
    x_cal = rng.uniform(0.0, 1.0, n)
    scale = 0.3 + 4.0 * x_cal
    r_cal = scale * np.abs(rng.normal(size=n))
    x_star = 0.9
    r_test = (0.3 + 4.0 * x_star) * np.abs(rng.normal(size=m_test))

    q_unif = weighted_conformal_quantile(r_cal, np.ones(n), ALPHA)
    w_ker = kernel_weights(x_cal[:, None], np.array([x_star]), bandwidth=0.08)
    q_ker = weighted_conformal_quantile(r_cal, w_ker, ALPHA)
    cov_unif = float(np.mean(r_test <= q_unif))
    cov_ker = float(np.mean(r_test <= q_ker))
    assert cov_ker >= 0.85, cov_ker
    assert cov_ker <= 0.96, cov_ker
    assert cov_unif < cov_ker - 0.03, (cov_unif, cov_ker)


def test_coverage_under_temporal_drift_synthetic() -> None:
    """SYNTHETIC: linearly growing volatility; recency weights track the drift.

    Correctness test only (seeded synthetic draws), not market evidence.
    """
    rng = np.random.default_rng(7)
    n, m_test = 400, 400
    s_cal = 0.5 + 3.0 * (np.arange(1, n + 1) / n)
    r_cal = s_cal * np.abs(rng.normal(size=n))
    s_test = 3.6  # drift continues past the calibration window
    r_test = s_test * np.abs(rng.normal(size=m_test))

    q_unif = weighted_conformal_quantile(r_cal, np.ones(n), ALPHA)
    q_rec = weighted_conformal_quantile(r_cal, recency_weights(n, bandwidth=40.0), ALPHA)
    cov_unif = float(np.mean(r_test <= q_unif))
    cov_rec = float(np.mean(r_test <= q_rec))
    assert cov_rec >= 0.85, cov_rec
    assert cov_rec <= 0.96, cov_rec
    assert cov_unif < cov_rec - 0.03, (cov_unif, cov_rec)


def test_empirical_coverage_lies_inside_theorem_bounds_synthetic() -> None:
    """SYNTHETIC: exchangeable Gaussian data, gap = 0 -> Theorems 2/3 bracket.

    With iid calibration/test residuals the true TV distances vanish, so the
    expected empirical coverage must sit inside [1 - alpha, 1 - alpha +
    w~_{n+1}] up to Monte-Carlo noise. Correctness test only, not market
    evidence.
    """
    n, m_test, n_seeds = 200, 300, 40
    w = recency_weights(n, bandwidth=25.0)
    _, w_test = normalize_weights(w)
    probe = _calibrate(np.abs(np.random.default_rng(0).normal(size=n)) + 0.01, w)
    bounds = probe.coverage_bounds(np.zeros(n))
    assert bounds.lower == pytest.approx(1.0 - ALPHA)
    assert bounds.upper == pytest.approx(1.0 - ALPHA + w_test)
    covs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        r_cal = np.abs(rng.normal(size=n)) + 0.01
        m = _calibrate(r_cal, w)
        r_test = np.abs(rng.normal(size=m_test)) + 0.01
        covs.append(float(np.mean(r_test <= m.quantile_)))
    cov = float(np.mean(covs))
    tol = 4.0 * float(np.std(covs, ddof=1)) / np.sqrt(n_seeds) + 0.01
    assert bounds.lower - tol <= cov <= bounds.upper + tol, (cov, bounds, tol)
    m = _calibrate(np.abs(np.random.default_rng(9).normal(size=n)) + 0.01, w)
    assert m.effective_sample_size_ == pytest.approx(float(w.sum()) + 1.0)


# ------------------------------------------------- NexCPSplit API and edges
def test_predict_interval_and_determinism() -> None:
    rng = np.random.default_rng(9)
    r = np.abs(rng.normal(size=120)) + 0.05
    w = kernel_weights(rng.uniform(size=(120, 2)), np.array([0.5, 0.5]), bandwidth=0.3)
    m1 = _calibrate(r, w)
    m2 = _calibrate(r.copy(), w.copy())
    assert m1.quantile_ == m2.quantile_
    np.testing.assert_array_equal(m1.weights_normalized_, m2.weights_normalized_)
    point = np.array([-1.0, 0.0, 2.5])
    lo, hi = m1.predict_interval(point)
    lo2, hi2 = m2.predict_interval(point)
    np.testing.assert_array_equal(lo, lo2)
    np.testing.assert_array_equal(hi, hi2)
    np.testing.assert_allclose(lo, point - m1.quantile_)
    np.testing.assert_allclose(hi, point + m1.quantile_)
    assert np.all(hi - lo == pytest.approx(2.0 * m1.quantile_))


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        NexCPSplit(alpha=0.0)
    with pytest.raises(ValueError):
        NexCPSplit(alpha=1.0)
    m = NexCPSplit(alpha=0.2)
    with pytest.raises(RuntimeError):
        m.predict_interval(np.zeros(2))
    with pytest.raises(RuntimeError):
        _ = m.quantile_
    with pytest.raises(RuntimeError):
        _ = m.weights_normalized_
    with pytest.raises(RuntimeError):
        _ = m.effective_sample_size_
    with pytest.raises(RuntimeError):
        m.coverage_bounds()
    r = np.abs(np.random.default_rng(1).normal(size=30)) + 0.05
    with pytest.raises(ValueError):
        m.calibrate(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        m.calibrate(r, np.ones(29))  # length mismatch
    with pytest.raises(ValueError):
        m.calibrate(-r, np.ones(30))  # negative absolute-error scores
    with pytest.raises(ValueError):
        m.calibrate(np.r_[np.nan, r[1:]], np.ones(30))
    with pytest.raises(ValueError):
        m.calibrate(r, np.full(30, np.inf))
    cal = m.calibrate(r, np.ones(30))
    with pytest.raises(ValueError):
        cal.predict_interval(np.array([]))
    with pytest.raises(ValueError):
        cal.predict_interval(np.array([np.nan]))
