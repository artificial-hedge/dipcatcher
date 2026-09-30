"""Tests for large_deviations: Gärtner-Ellis rate functions, Chernoff bounds,
exponential-tilting importance sampling, and subadditivity for portfolio risk.

All Monte Carlo draws use seeded generators — determinism pinned.
All VaR/ES numbers are SYNTHETIC — correctness evidence, never market data.
MC sample sizes are kept modest for test speed (≤ 50k per test).

Coverage:
  (1) Rate-function closed forms: Gaussian (quadratic) and independent
      Exponential (log-sum) — analytic expectations pinned at test points.
  (2) Chernoff bound: MC tail ≤ exp(−Λ*(t)) + noise.
  (3) IS variance reduction: Var(crude) / Var(IS) > 2.
  (4) Subadditivity of the rate function for independent sums (Gaussian
      and exponential).
  (5) Fail-closed edges: non-finite mgf, t outside support, domain violations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.large_deviations import (
    chernoff_bound,
    exponential_log_mgf,
    exponential_log_mgf_grad,
    exponential_rate,
    factor_model_covariance,
    factor_model_is_tail_prob,
    factor_model_sample,
    fenchel_legendre,
    gaussian_is_tail_prob,
    gaussian_log_mgf,
    gaussian_log_mgf_grad,
    gaussian_rate,
    importance_sampling_report,
    optimal_tilt_gaussian,
    rate_subadditivity,
)

Array = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Pinned constants — determinism + documented tolerances
# ---------------------------------------------------------------------------

SEED = 20260929
MC_TOL = 0.02  # binomial slack for empirical-tail ≤ bound checks
MC_N = 50_000  # modest sample size for Chernoff coverage
IS_CRUDE_N = 25_000
IS_N = 8_000


# ---------------------------------------------------------------------------
# (1) Gärtner-Ellis rate-function closed forms
# ---------------------------------------------------------------------------


def test_gaussian_rate_quadratic_form_diagonal() -> None:
    """Λ*(x) = ½(x−μ)ᵀ Σ⁻¹(x−μ) — pinned against analytic expectation.

    μ = [1, 1],  Σ = diag(4, 9).  At x = [3, 4]:
      diff = [2, 3],  Σ⁻¹ diff = [2/4, 3/9] = [0.5, 0.333...]
      Λ* = ½(2·0.5 + 3·0.333...) = ½(1 + 1) = 1.0.
    """
    mu = np.array([1.0, 1.0])
    cov = np.array([[4.0, 0.0], [0.0, 9.0]])
    rate = gaussian_rate(np.array([3.0, 4.0]), mu, cov)
    assert rate == pytest.approx(1.0, rel=1e-12)


def test_gaussian_rate_quadratic_form_correlated() -> None:
    """Λ*(x) with a non-diagonal Σ against manually computed expectation.

    μ = [0, 0],  Σ = [[2, 0.5], [0.5, 1]].  At x = [1, 1]:
      Σ⁻¹ = 1/(2−0.25) [[1, −0.5], [−0.5, 2]] = [[0.5714…, −0.2857…],
                                                   [−0.2857…, 1.1428…]]
      diff = [1, 1]
      Λ* = ½ [1, 1] Σ⁻¹ [1, 1]ᵀ = ½·(1·1·0.5714 + 1·1·1.1428 − 2·1·1·0.2857)
         = ½·(0.5714 + 1.1428 − 0.5714) = ½·1.1428… ≈ 0.571428…
    """
    mu = np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    rate = gaussian_rate(np.array([1.0, 1.0]), mu, cov)
    # det = 1.75, Σ⁻¹ = [[1/1.75, −0.5/1.75], [−0.5/1.75, 2/1.75]]
    # [1,1] Σ⁻¹ [1,1] = (1 − 0.5 − 0.5 + 2)/1.75 = 2/1.75 = 8/7
    expected = 0.5 * (2.0 / 1.75)  # = 1/1.75 = 4/7 ≈ 0.5714285714285714
    assert rate == pytest.approx(expected, rel=1e-12)


def test_gaussian_rate_at_mean_is_zero() -> None:
    """Λ*(μ) = 0 for any μ, Σ."""
    mu = np.array([1.5, -0.7, 2.1])
    cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.3], [0.1, 0.3, 3.0]])
    assert gaussian_rate(mu, mu, cov) == pytest.approx(0.0, abs=1e-12)


def test_exponential_rate_log_sum_form() -> None:
    """Λ*(x) = Σᵢ(rateᵢ·xᵢ − 1 − log(rateᵢ·xᵢ)) — pinned analytic values.

    rates = [2, 3],  x = [1.0, 1.0]:
      comp1: 2·1 − 1 − log(2) = 1 − log 2
      comp2: 3·1 − 1 − log(3) = 2 − log 3
      total = 3 − log 6.
    """
    rates = np.array([2.0, 3.0])
    rate = exponential_rate(np.array([1.0, 1.0]), rates)
    expected = 3.0 - math.log(6.0)
    assert rate == pytest.approx(expected, rel=1e-12)


def test_exponential_rate_at_mean_is_zero() -> None:
    """Λ*(x) = 0 at x = 1/rate (the mean of an Exponential)."""
    r = 2.5
    x_mean = np.array([1.0 / r])
    rate = exponential_rate(x_mean, np.array([r]))
    assert rate == pytest.approx(0.0, abs=1e-10)


def test_exponential_rate_non_positive_is_inf() -> None:
    """xᵢ ≤ 0 ⇒ Λ*(x) = +∞ (outside support)."""
    assert exponential_rate(np.array([0.0]), np.array([1.0])) == float("inf")
    assert exponential_rate(np.array([-0.3]), np.array([2.0])) == float("inf")
    assert exponential_rate(np.array([0.5, 0.0]), np.array([1.0, 2.0])) == float("inf")


def test_gaussian_rate_matches_numerical_fenchel_legendre() -> None:
    """Numerical Fenchel-Legendre recovers the closed-form Gaussian rate."""
    mu = np.array([0.1, -0.3])
    cov = np.array([[1.5, 0.4], [0.4, 2.0]])
    for x in [
        np.array([0.5, -0.1]),
        np.array([-0.3, 0.7]),
        np.array([1.2, -0.8]),
        np.array([0.0, 0.0]),
    ]:
        closed = gaussian_rate(x, mu, cov)
        numerical = fenchel_legendre(
            x,
            log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
            grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
        )
        assert closed == pytest.approx(numerical, abs=1e-6), f"mismatch at x={x}"


def test_exponential_rate_matches_numerical_fenchel_legendre() -> None:
    """Numerical Fenchel-Legendre recovers the closed-form exponential rate."""
    rates = np.array([1.5, 2.5])
    for x in [np.array([0.4, 0.3]), np.array([1.2, 0.7]), np.array([0.3, 0.15])]:
        closed = exponential_rate(x, rates)
        numerical = fenchel_legendre(
            x,
            log_mgf=lambda lam: exponential_log_mgf(lam, rates),
            grad_log_mgf=lambda lam: exponential_log_mgf_grad(lam, rates),
            lam_bounds=(-1e4, float(np.min(rates)) - 1e-6),
        )
        assert closed == pytest.approx(numerical, abs=1e-5), f"mismatch at x={x}"


# ---------------------------------------------------------------------------
# (2) Chernoff bound: empirical tail ≤ exp(−Λ*(t)) + MC noise
# ---------------------------------------------------------------------------


def test_chernoff_bound_gaussian_mc() -> None:
    """P(w·X ≥ t) ≤ exp(−Λ*(t)) for scalar Gaussian, verified via MC.

    SYNTHETIC — 50k paths.  For N(0, 1): bound = exp(−½ t²).
    """
    rng = np.random.default_rng(SEED)
    mu = np.array([0.0])
    cov = np.array([[1.0]])
    w = np.array([1.0])

    samples = rng.multivariate_normal(mu, cov, size=MC_N)
    losses = np.asarray(samples @ w, dtype=np.float64).ravel()

    for t in [1.5, 2.0, 2.5]:
        emp = float(np.mean(losses >= t))
        bound = chernoff_bound(
            t,
            w,
            log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
            grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
        )
        assert emp <= bound + MC_TOL, (
            f"Chernoff violated at t={t}: empirical={emp:.6g}, bound={bound:.6g}"
        )
        assert bound < 1.0, f"Chernoff uninformative at t={t}"


def test_chernoff_bound_exponential_mc() -> None:
    """P(X ≥ t) ≤ exp(−(r·t − 1 − log(r·t))) for Exp(r) with t > 1/r.

    SYNTHETIC — 50k paths.
    """
    rng = np.random.default_rng(SEED + 1)
    rate = 2.0
    w = np.array([1.0])

    # Draw Exponential(rate) samples
    samples = rng.exponential(1.0 / rate, size=MC_N)
    losses = np.asarray(samples * w[0], dtype=np.float64)  # shape (MC_N,)

    for t in [1.0, 1.5, 2.0]:  # all > mean = 0.5
        emp = float(np.mean(losses >= t))
        bound = chernoff_bound(
            t,
            w,
            log_mgf=lambda lam: exponential_log_mgf(lam, np.array([rate])),
            grad_log_mgf=lambda lam: exponential_log_mgf_grad(lam, np.array([rate])),
            theta_max=rate * 0.999,  # stay inside domain
        )
        assert emp <= bound + MC_TOL, (
            f"Chernoff violated at t={t}: empirical={emp:.6g}, bound={bound:.6g}"
        )
        assert bound < 1.0, f"Chernoff uninformative at t={t}"


def test_chernoff_bound_multivariate_gaussian_mc() -> None:
    """Chernoff holds for a 3-d Gaussian portfolio with non-uniform weights.

    SYNTHETIC — 50k paths.
    """
    rng = np.random.default_rng(SEED + 2)
    mu = np.array([0.1, -0.05, 0.2])
    cov = np.array([[1.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 2.0]])
    w = np.array([0.5, 1.0, -0.3])

    samples = rng.multivariate_normal(mu, cov, size=MC_N)
    losses = np.asarray(samples @ w, dtype=np.float64)

    sigma = math.sqrt(float(w @ (cov @ w)))
    for z in [2.0, 2.5, 3.0]:
        t = float(np.dot(w, mu)) + z * sigma
        emp = float(np.mean(losses >= t))
        bound = chernoff_bound(
            t,
            w,
            log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
            grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
        )
        assert emp <= bound + MC_TOL, (
            f"Chernoff violated at z={z}: empirical={emp:.6g}, bound={bound:.6g}"
        )
        assert bound < 1.0, f"Chernoff uninformative at z={z}"


def test_chernoff_bound_below_mean_returns_one() -> None:
    """When t ≤ E[w·X], the Chernoff bound is 1.0 (uninformative)."""
    mu = np.array([2.0])
    cov = np.array([[1.0]])
    w = np.array([1.0])
    bound = chernoff_bound(
        1.0,  # t < w·μ = 2
        w,
        log_mgf=lambda lam: gaussian_log_mgf(lam, mu, cov),
        grad_log_mgf=lambda lam: gaussian_log_mgf_grad(lam, mu, cov),
    )
    assert bound == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# (3) IS variance reduction: Var(crude) / Var(IS) > 2
# ---------------------------------------------------------------------------


def test_gaussian_is_vrf_gt_two() -> None:
    """Exponential-tilting IS on a Gaussian linearized portfolio: VRF > 2.

    SYNTHETIC — 25k crude + 8k IS.  Threshold ~2σ in the right tail.
    """
    rng = np.random.default_rng(SEED + 3)
    mu = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])
    w = np.array([0.6, 0.6, 0.4])  # portfolio weights

    sigma = math.sqrt(float(w @ (cov @ w)))
    threshold = 2.0 * sigma  # ≈ 2σ tail event, P ≈ 0.023

    result = gaussian_is_tail_prob(
        rng, n_is=IS_N, n_crude=IS_CRUDE_N, w=w, mu=mu, cov=cov, threshold=threshold
    )

    assert result["synthetic"] is True
    assert result["crude_estimate"] >= 0.0
    assert result["is_estimate"] >= 0.0

    vrf = result["variance_reduction_factor"]
    assert vrf is not None, f"VRF is None: {result.get('reason')}"
    assert vrf > 2.0, (
        f"IS variance reduction too low: VRF={vrf:.2f}, "
        f"crude_var={result['crude_variance']:.2g}, "
        f"is_var={result['is_variance']:.2g}"
    )


def test_gaussian_is_estimates_converge_to_true_prob() -> None:
    """Both crude and IS estimates are within MC noise of the true value.

    SYNTHETIC — 50k paths each.  True tail probability from standard normal.
    """
    rng = np.random.default_rng(SEED + 4)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([1.0, 0.5])  # w·X ~ N(0, 1.25), σ = √1.25 ≈ 1.118

    sigma = math.sqrt(float(w @ (cov @ w)))
    threshold = 2.0 * sigma  # 2σ
    true_prob = float(1.0 - 0.9772498680518208)  # Φ(−2) ≈ 0.02275

    result = gaussian_is_tail_prob(
        rng, n_is=50_000, n_crude=50_000, w=w, mu=mu, cov=cov, threshold=threshold
    )

    assert result["crude_estimate"] == pytest.approx(true_prob, abs=0.008)
    assert result["is_estimate"] == pytest.approx(true_prob, abs=0.005)


def test_gaussian_is_vrf_increases_with_rarity() -> None:
    """IS becomes more effective (higher VRF) as the tail event gets rarer.

    SYNTHETIC — properties of exponential tilting under Gaussian model.
    """
    rng = np.random.default_rng(SEED + 5)
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    w = np.array([1.0, 1.0])
    sigma = math.sqrt(float(w @ (cov @ w)))

    vrfs: list[float] = []
    for z in [1.5, 2.5]:
        result = gaussian_is_tail_prob(
            rng,
            n_is=IS_N,
            n_crude=IS_CRUDE_N,
            w=w,
            mu=mu,
            cov=cov,
            threshold=z * sigma,
        )
        vrf = result["variance_reduction_factor"]
        assert vrf is not None, f"VRF is None at z={z}: {result.get('reason')}"
        assert vrf > 1.0, f"VRF={vrf:.2f} not >1 at z={z}"
        vrfs.append(vrf)

    # At z=2.5 the event is rarer → IS should give larger relative improvement.
    assert vrfs[1] > vrfs[0], (
        f"VRF should increase with rarity: VRF(z=1.5)={vrfs[0]:.2f}, VRF(z=2.5)={vrfs[1]:.2f}"
    )


# ---------------------------------------------------------------------------
# (4) Subadditivity of the rate function for independent sums
# ---------------------------------------------------------------------------


def _scalar_gaussian_rate(t: float, mean: float, var: float) -> float:
    """Scalar Gaussian rate: Λ*(t) = (t−μ)² / (2σ²)."""
    d = t - mean
    return d * d / (2.0 * var)


class TestSubadditivityGaussian:
    """Rate-function subadditivity for independent Gaussian components."""

    def test_independent_equal_variance(self) -> None:
        """X₁, X₂ i.i.d. N(μ, σ²) → X₁+X₂ ~ N(2μ, 2σ²). Subadditivity tight."""
        mu_x, var_x = 0.5, 1.0
        mu_sum = 2.0 * mu_x
        var_sum = 2.0 * var_x

        def rate_sum(t_arr: Array) -> float:
            return _scalar_gaussian_rate(float(t_arr[0]), mu_sum, var_sum)

        def rate_x(t_arr: Array) -> float:
            return _scalar_gaussian_rate(float(t_arr[0]), mu_x, var_x)

        t_vals = np.linspace(0.5, 6.0, 30)
        result = rate_subadditivity(rate_sum, rate_x, rate_x, t_vals)

        assert result["passed"], (
            f"Subadditivity violated: {result['n_violations']} violations, "
            f"max={result['max_violation']:.2g}"
        )
        assert result["n_violations"] == 0

    def test_independent_unequal_variance(self) -> None:
        """X₁ ~ N(μ₁, σ²₁), X₂ ~ N(μ₂, σ²₂), independent → subadditivity holds."""
        mu1, var1 = 1.0, 2.0
        mu2, var2 = 0.5, 1.5

        def rate_sum(t_arr: Array) -> float:
            return _scalar_gaussian_rate(float(t_arr[0]), mu1 + mu2, var1 + var2)

        def rate_x(t_arr: Array) -> float:
            return _scalar_gaussian_rate(float(t_arr[0]), mu1, var1)

        def rate_y(t_arr: Array) -> float:
            return _scalar_gaussian_rate(float(t_arr[0]), mu2, var2)

        t_vals = np.linspace(0.5, 7.0, 35)
        result = rate_subadditivity(rate_sum, rate_x, rate_y, t_vals)

        assert result["passed"]
        assert result["n_violations"] == 0


class TestSubadditivityExponential:
    """Rate-function subadditivity for independent Exponential components."""

    @staticmethod
    def _gamma_rate(t: float, lam_rate: float, k: float) -> float:
        """Rate function for Gamma(k, 1/λ): k·(λ·t/k − 1 − log(λ·t/k))."""
        if t <= 0:
            return float("inf")
        return lam_rate * t - k - k * math.log(lam_rate * t / k)

    def test_iid_components(self) -> None:
        """Two i.i.d. Exp(λ) → sum ~ Gamma(2, 1/λ). Subadditivity holds."""
        lam_rate = 1.5

        def rate_x(t_arr: Array) -> float:
            return exponential_rate(t_arr, np.array([lam_rate]))

        def rate_sum(t_arr: Array) -> float:
            return self._gamma_rate(float(t_arr[0]), lam_rate, 2.0)

        t_vals = np.linspace(0.2, 5.0, 30)
        result = rate_subadditivity(rate_sum, rate_x, rate_x, t_vals)

        assert result["passed"], (
            f"Subadditivity violated: {result['n_violations']} violations, "
            f"max={result['max_violation']:.2g}"
        )
        assert result["n_violations"] == 0

    def test_different_rates(self) -> None:
        """X₁ ~ Exp(λ₁), X₂ ~ Exp(λ₂), independent. The sum rate is computed
        via the joint log-MGF and Fenchel-Legendre; subadditivity holds."""
        r1, r2 = 2.0, 3.0

        def rate_x(t_arr: Array) -> float:
            return exponential_rate(t_arr, np.array([r1]))

        def rate_y(t_arr: Array) -> float:
            return exponential_rate(t_arr, np.array([r2]))

        def rate_sum(t_arr: Array) -> float:
            """Rate of the sum X₁+X₂ via numerical Fenchel-Legendre of the
            joint log-MGF: Λ(θ) = −log(1−θ/r₁) − log(1−θ/r₂)."""
            t_val = float(t_arr[0])
            if t_val <= 0:
                return float("inf")
            r_min = min(r1, r2)

            def joint_log_mgf(lam: Array) -> float:
                theta = float(lam[0])
                return -math.log(1.0 - theta / r1) - math.log(1.0 - theta / r2)

            def joint_grad(lam: Array) -> Array:
                theta = float(lam[0])
                return np.array([1.0 / (r1 - theta) + 1.0 / (r2 - theta)])

            return fenchel_legendre(
                t_arr,
                joint_log_mgf,
                joint_grad,
                lam_bounds=(-1e4, r_min - 1e-6),
            )

        t_vals = np.linspace(0.3, 4.0, 25)
        result = rate_subadditivity(rate_sum, rate_x, rate_y, t_vals)

        assert result["passed"], (
            f"Subadditivity violated: {result['n_violations']} violations, "
            f"max={result['max_violation']:.2g}"
        )
        assert result["n_violations"] == 0


# ---------------------------------------------------------------------------
# (5) Fail-closed edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor",
    [
        # --- log-MGF domain violations ---
        lambda: gaussian_log_mgf(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0]), np.eye(2)),
        lambda: gaussian_log_mgf(np.array([np.nan]), np.array([1.0]), np.eye(1)),
        lambda: gaussian_log_mgf(
            np.array([1.0]), np.array([1.0]), np.array([[1.0, 1.0], [1.0, 1.0]])
        ),
        lambda: exponential_log_mgf(np.array([1.5]), np.array([1.0])),
        lambda: exponential_log_mgf(np.array([0.1]), np.array([0.0])),
        lambda: exponential_log_mgf(np.array([0.1, 0.2]), np.array([1.0])),
        # --- rate-function domain violations ---
        lambda: gaussian_rate(np.array([1.0]), np.array([0.0, 0.0]), np.eye(2)),
        lambda: gaussian_rate(
            np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 0.0]])
        ),
        lambda: exponential_rate(np.array([1.0]), np.array([0.0])),
        lambda: exponential_rate(np.array([1.0, 2.0]), np.array([1.0])),
        # --- Chernoff bound ---
        lambda: chernoff_bound(float("inf"), np.array([1.0]), log_mgf=lambda lam: 0.0),
        lambda: chernoff_bound(float("nan"), np.array([1.0]), log_mgf=lambda lam: 0.0),
        # --- optimal tilt ---
        lambda: optimal_tilt_gaussian(1.0, np.array([1.0, 2.0]), np.array([0.0]), np.eye(1)),
        lambda: optimal_tilt_gaussian(float("nan"), np.array([1.0]), np.array([0.0]), np.eye(1)),
        lambda: optimal_tilt_gaussian(
            1.0, np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.zeros((2, 2))
        ),
        # --- Gaussian IS ---
        lambda: gaussian_is_tail_prob(
            np.random.default_rng(0),
            n_is=1,
            n_crude=100,
            w=np.array([1.0]),
            mu=np.array([0.0]),
            cov=np.eye(1),
            threshold=1.0,
        ),
        lambda: gaussian_is_tail_prob(
            np.random.default_rng(0),
            n_is=100,
            n_crude=1,
            w=np.array([1.0]),
            mu=np.array([0.0]),
            cov=np.eye(1),
            threshold=1.0,
        ),
        lambda: gaussian_is_tail_prob(
            np.random.default_rng(0),
            n_is=100,
            n_crude=100,
            w=np.array([1.0, 2.0]),
            mu=np.array([0.0]),
            cov=np.eye(1),
            threshold=1.0,
        ),
        # --- Fenchel-Legendre ---
        lambda: fenchel_legendre(np.array([]), lambda lam: 0.0),
        lambda: fenchel_legendre(
            np.array([1.0, 2.0]),
            lambda lam: 0.0,
            lam0=np.array([1.0]),
        ),
        # --- factor model ---
        lambda: factor_model_sample(
            np.random.default_rng(0),
            n=100,
            w=np.array([1.0]),
            betas=np.array([[0.5]]),
            psi=np.array([0.75]),
            df=2.0,
        ),
        lambda: factor_model_covariance(np.array([[1.0]]), np.array([-0.5])),
        lambda: factor_model_covariance(np.array([[1.0]]), np.array([0.5, 0.6])),
        # --- factor model IS edge ---
        lambda: factor_model_is_tail_prob(
            np.random.default_rng(0),
            n_is=100,
            n_crude=100,
            w=np.array([1.0, 2.0]),
            betas=np.array([[0.5]]),
            psi=np.array([0.75]),
            threshold=1.0,
            df=4.0,
        ),
    ],
)
def test_fail_closed_raises_value_error(ctor) -> None:
    """All invalid inputs raise ValueError (fail-closed)."""
    with pytest.raises(ValueError):
        ctor()


def test_synthetic_marker_on_all_public_results() -> None:
    """Every public result dict carries ``synthetic=True``."""
    mu = np.array([0.0])
    cov = np.eye(1)
    w = np.array([1.0])

    r1 = gaussian_is_tail_prob(np.random.default_rng(0), 500, 500, w, mu, cov, 2.0)
    assert r1["synthetic"] is True
    assert r1["method"] == "gaussian_exponential_tilting_is"

    r2 = importance_sampling_report(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.2, 0.3]))
    assert r2["synthetic"] is True
    assert r2["method"] == "importance_sampling_generic"


# ---------------------------------------------------------------------------
# Determinism: seeded results are bit-identical
# ---------------------------------------------------------------------------


def test_determinism_gaussian_is() -> None:
    """Same seed → identical IS results (determinism pinned)."""
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    w = np.array([1.0, 1.0])

    r1 = gaussian_is_tail_prob(np.random.default_rng(123), 5000, 5000, w, mu, cov, 3.0)
    r2 = gaussian_is_tail_prob(np.random.default_rng(123), 5000, 5000, w, mu, cov, 3.0)

    assert r1["is_estimate"] == r2["is_estimate"]
    assert r1["crude_estimate"] == r2["crude_estimate"]
    assert r1["variance_reduction_factor"] == r2["variance_reduction_factor"]


def test_determinism_factor_model_is() -> None:
    """Same seed → identical factor-model IS results."""
    betas = np.array([[0.5], [0.5]], dtype=np.float64)
    psi = np.array([0.75, 0.75], dtype=np.float64)
    w = np.array([1.0, 1.0], dtype=np.float64)

    r1 = factor_model_is_tail_prob(
        np.random.default_rng(456),
        2000,
        2000,
        w,
        betas,
        psi,
        1.5,
        df=4.0,
    )
    r2 = factor_model_is_tail_prob(
        np.random.default_rng(456),
        2000,
        2000,
        w,
        betas,
        psi,
        1.5,
        df=4.0,
    )
    assert r1["is_estimate"] == r2["is_estimate"]
    assert r1["crude_estimate"] == r2["crude_estimate"]


# ---------------------------------------------------------------------------
# Log-MGF and gradient sanity checks
# ---------------------------------------------------------------------------


def test_gaussian_log_mgf_linear_plus_quadratic() -> None:
    """Λ(λ) = μ·λ + ½ λᵀ Σ λ."""
    mu = np.array([0.2, 0.8])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    lam = np.array([0.5, -0.3])
    expected = float(np.dot(lam, mu) + 0.5 * (lam @ (cov @ lam)))
    assert gaussian_log_mgf(lam, mu, cov) == pytest.approx(expected, abs=1e-12)


def test_gaussian_log_mgf_gradient() -> None:
    """∇Λ(λ) = μ + Σ λ."""
    mu = np.array([0.0, 1.0])
    cov = np.array([[2.0, 0.0], [0.0, 3.0]])
    lam = np.array([0.4, -0.2])
    grad = gaussian_log_mgf_grad(lam, mu, cov)
    expected = mu + cov @ lam
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_exponential_log_mgf_gradient_known() -> None:
    """∂Λ/∂λᵢ = 1/(rateᵢ − λᵢ)."""
    rates = np.array([1.0, 3.0])
    lam = np.array([0.3, 1.5])
    grad = exponential_log_mgf_grad(lam, rates)
    expected = np.array([1.0 / (1.0 - 0.3), 1.0 / (3.0 - 1.5)])
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_optimal_tilt_puts_mean_at_threshold() -> None:
    """θ* satisfies w·(μ + θ Σ w) = threshold (Gaussian closed form)."""
    mu = np.array([0.5, -0.2])
    cov = np.array([[2.0, 0.1], [0.1, 1.5]])
    w = np.array([1.0, 0.8])
    threshold = 4.0

    theta = optimal_tilt_gaussian(threshold, w, mu, cov)
    tilted_mean = float(np.dot(w, mu)) + theta * float(w @ (cov @ w))
    assert tilted_mean == pytest.approx(threshold, abs=1e-10)


def test_optimal_tilt_below_mean_returns_zero() -> None:
    """When t ≤ w·μ, no tilt is needed (θ = 0)."""
    mu = np.array([5.0])
    cov = np.array([[1.0]])
    w = np.array([1.0])
    assert optimal_tilt_gaussian(3.0, w, mu, cov) == 0.0
