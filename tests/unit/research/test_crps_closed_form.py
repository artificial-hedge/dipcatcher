"""DayWave7: closed-form Gaussian CRPS + empirical ensemble CRPS.

Research-diagnostic only — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.scoring import (
    crps_empirical,
    crps_gaussian,
    crps_student_t,
    mean_crps_gaussian,
    mean_crps_student_t,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_crps_gaussian_standard_normal_at_median() -> None:
    """μ=0, σ=1, y=0 → σ[2φ(0) − 1/√π] = (√2 − 1)/√π (not 1/√π)."""
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    out = crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert out[0] == pytest.approx(expected)
    assert mean_crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0])) == pytest.approx(
        expected
    )


def test_crps_gaussian_scale_homogeneity() -> None:
    """CRPS(N(μ,σ²), y) = σ · CRPS(N(0,1), (y−μ)/σ)."""
    y, mu, sigma = 2.0, 1.0, 3.0
    z = (y - mu) / sigma
    unit = crps_gaussian(np.array([z]), np.array([0.0]), np.array([1.0]))[0]
    scaled = crps_gaussian(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    assert scaled == pytest.approx(sigma * unit)


def test_crps_gaussian_empty_mismatch_bad_sigma() -> None:
    assert crps_gaussian(np.array([]), np.array([]), np.array([])).size == 0
    assert np.isnan(mean_crps_gaussian(np.array([]), np.array([]), np.array([])))
    with pytest.raises(ValueError, match="length mismatch"):
        crps_gaussian(np.array([0.0, 1.0]), np.array([0.0]), np.array([1.0]))
    # Non-positive / non-finite σ → NaN (fail-closed honesty, not ValueError)
    out = crps_gaussian(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, -1.0, np.nan]),
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(mean_crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([0.0])))


def test_crps_from_quantiles_matches_closed_form_gaussian() -> None:
    """Dense-tau Riemann sum must converge to the closed-form CRPS (factor 2).

    Regression: the quantile decomposition was missing the Gneiting–Raftery
    factor 2, silently reporting half-CRPS beside the closed-form keys.
    """
    from scipy.stats import norm

    from quant_fund.metrics.scoring import crps_from_quantiles

    taus = np.linspace(0.001, 0.999, 2000)
    y = np.array([0.3])
    mu = np.array([0.0])
    sigma = np.array([1.0])
    q = mu[:, None] + sigma[:, None] * norm.ppf(taus)[None, :]
    assert crps_from_quantiles(y, q, taus) == pytest.approx(
        crps_gaussian(y, mu, sigma)[0], rel=1e-3
    )


def test_crps_empirical_matches_two_point_ensemble() -> None:
    """Hand-check: y=0, sample={-1, 1}.

    term1 = mean(|Xi−y|) = 1
    term2 = (1/(2n²)) ΣΣ|Xi−Xj| = (1/8)*(0+2+2+0) = 0.5
    CRPS = 1 − 0.5 = 0.5
    """
    assert crps_empirical(0.0, np.array([-1.0, 1.0])) == pytest.approx(0.5)
    assert crps_empirical(np.array([0.0]), np.array([-1.0, 1.0])) == pytest.approx(0.5)


def test_crps_empirical_gaussian_ensemble_near_closed_form() -> None:
    """Large N(0,1) ensemble at y=0 ≈ closed-form (√2−1)/√π."""
    rng = np.random.default_rng(7)
    sample = rng.normal(0.0, 1.0, size=20_000)
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert crps_empirical(0.0, sample) == pytest.approx(expected, abs=0.01)


def test_crps_empirical_edges() -> None:
    assert np.isnan(crps_empirical(0.0, np.array([])))
    assert np.isnan(crps_empirical(0.0, np.array([np.nan, np.nan])))
    assert np.isnan(crps_empirical(np.nan, np.array([0.0, 1.0])))
    with pytest.raises(ValueError, match="single observation"):
        crps_empirical(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# --- DayWave43: closed-form Student-t CRPS ---


def test_crps_student_t_median_nu5() -> None:
    """μ=0, σ=1, ν=5, y=0 → 2/(ν−1)·(f(0)·ν − √ν·bfrac)."""
    from scipy.special import betaln
    from scipy.stats import t as student_t

    nu = 5.0
    f0 = float(student_t.pdf(0.0, nu))
    bfrac = math.exp(betaln(0.5, nu - 0.5) - 2.0 * betaln(0.5, 0.5 * nu))
    expected = 2.0 / (nu - 1.0) * (f0 * nu - math.sqrt(nu) * bfrac)
    out = crps_student_t(np.array([0.0]), np.array([0.0]), np.array([1.0]), nu)
    assert out[0] == pytest.approx(expected)
    assert mean_crps_student_t(
        np.array([0.0]), np.array([0.0]), np.array([1.0]), nu
    ) == pytest.approx(expected)


def test_crps_student_t_scale_homogeneity() -> None:
    """CRPS(t_{μ,σ,ν}, y) = σ · CRPS(t_{0,1,ν}, (y−μ)/σ)."""
    y, mu, sigma, nu = 2.0, 1.0, 3.0, 8.0
    z = (y - mu) / sigma
    unit = crps_student_t(np.array([z]), np.array([0.0]), np.array([1.0]), nu)[0]
    scaled = crps_student_t(np.array([y]), np.array([mu]), np.array([sigma]), nu)[0]
    assert scaled == pytest.approx(sigma * unit)


def test_crps_student_t_large_nu_near_gaussian() -> None:
    """Large ν ≈ Gaussian CRPS (loose abs tol)."""
    y = np.array([0.0, 0.5, -1.2])
    mu = np.array([0.0, 0.0, 0.1])
    sigma = np.array([1.0, 1.5, 0.8])
    t_scores = crps_student_t(y, mu, sigma, 1e6)
    g_scores = crps_gaussian(y, mu, sigma)
    assert t_scores == pytest.approx(g_scores, abs=1e-4)


def _scipy_stats_unit_crps(z: np.ndarray, nu: float) -> np.ndarray:
    """Independent scoringRules evaluation via ``scipy.stats.t``, not ``stdtr``."""
    from scipy.special import betaln
    from scipy.stats import t as student_t

    bfrac = math.exp(betaln(0.5, nu - 0.5) - 2.0 * betaln(0.5, 0.5 * nu))
    density = student_t.pdf(z, nu)
    cdf = student_t.cdf(z, nu)
    return z * (2.0 * cdf - 1.0) + (2.0 / (nu - 1.0)) * (
        density * (nu + z * z) - math.sqrt(nu) * bfrac
    )


def test_crps_student_t_matches_scipy_stats_reference() -> None:
    """Closed-form kernel matches ``scipy.stats.t`` pdf/cdf on a seeded grid."""
    rng = np.random.default_rng(11)
    y = rng.normal(size=64)
    mu = rng.normal(scale=0.2, size=64)
    sigma = rng.uniform(0.2, 3.0, size=64)
    for nu in (2.05, 3.0, 5.0, 8.0, 30.0, 100.0):
        got = crps_student_t(y, mu, sigma, nu)
        expect = sigma * _scipy_stats_unit_crps((y - mu) / sigma, nu)
        assert got == pytest.approx(expect, rel=1e-10, abs=1e-12)


def test_crps_student_t_varying_nu_matches_scalar_calls() -> None:
    y = np.array([0.0, 0.5, -1.0, 2.0])
    mu = np.zeros(4)
    sigma = np.ones(4)
    nu = np.array([3.0, 5.0, 8.0, 20.0])
    got = crps_student_t(y, mu, sigma, nu)
    expect = np.array(
        [
            crps_student_t(y[i : i + 1], mu[i : i + 1], sigma[i : i + 1], float(nu[i]))[0]
            for i in range(4)
        ]
    )
    assert got == pytest.approx(expect, rel=1e-12, abs=1e-12)


def test_crps_student_t_constant_nu_array_matches_scalar() -> None:
    y = np.linspace(-2.0, 2.0, 17)
    mu = np.zeros_like(y)
    sigma = np.full_like(y, 1.5)
    scalar = crps_student_t(y, mu, sigma, 5.0)
    vector = crps_student_t(y, mu, sigma, np.full(y.shape, 5.0))
    assert vector == pytest.approx(scalar, rel=1e-12, abs=1e-12)


def test_crps_student_t_empty_mismatch_bad_sigma_nu() -> None:
    assert crps_student_t(np.array([]), np.array([]), np.array([]), 5.0).size == 0
    assert np.isnan(mean_crps_student_t(np.array([]), np.array([]), np.array([]), 5.0))
    with pytest.raises(ValueError, match="length mismatch"):
        crps_student_t(np.array([0.0, 1.0]), np.array([0.0]), np.array([1.0]), 5.0)
    # Non-positive / non-finite σ → NaN; ν≤2 → NaN (fail-closed)
    out = crps_student_t(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, -1.0, np.nan, 1.0]),
        np.array([5.0, 5.0, 5.0, 5.0, 2.0]),
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])
    assert np.isnan(mean_crps_student_t(np.array([0.0]), np.array([0.0]), np.array([1.0]), 1.5))
