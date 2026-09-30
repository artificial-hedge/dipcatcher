"""Tests for confidence_sequences: anytime-valid CSs for stream means.

Constructions under test:
- sub-Gaussian mixture / poly-stitching / poly-hedge-epsilon CS
  (Howard, Ramdas, McAuliffe & Sekhon 2021, arXiv:1810.08240; the
  hedge-epsilon piece from Jamieson et al. 2014, arXiv:1405.3269);
- betting (WSR) hedged-capital CS for bounded data (Waudby-Smith & Ramdas
  2024, arXiv:2010.09686, Theorem 3);
- empirical-Bernstein CS via stitching (Howard et al. 2021, Theorem 4).

All streams are seeded SYNTHETIC draws (np.random.default_rng with pinned
seeds — determinism, no market data, no market evidence). Monte-Carlo
tolerances: with R replicates at level alpha the empirical time-uniform
violation rate must stay <= alpha + 0.02 (binomial slack, the convention in
test_e_detectors.py); per-time marginal coverage must stay >= 1 - alpha -
0.03. Heavy tails use a Student-t(3) stream symmetrically clipped to
[-8, 8] — the documented relaxation: the clip preserves the mean (symmetry)
and restores the bounded support that WSR / empirical-Bernstein require;
unclipped t3 admits no finite sub-Gaussian proxy variance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.confidence_sequences import (
    ConfidenceSequence,
    check_time_uniform_coverage,
    cs_widths,
    empirical_bernstein_cs,
    fixed_sample_ci_width,
    hedge_epsilon_boundary,
    normal_mixture_boundary,
    optimal_mixture_rho,
    poly_hedge_epsilon_boundary,
    poly_stitching_boundary,
    subgaussian_cs,
    width_decay_slope,
    width_ratio_vs_fixed,
    wsr_cs,
)

Array = NDArray[np.float64]

SEED = 20260928
ALPHA = 0.05
# Documented MC tolerances (binomial slack at ~100-200 replicates).
VIOL_TOL = 0.02
PERTIME_TOL = 0.03
N_REPS = 200
T_GAUSS = 1000
N_REPS_WSR = 80
T_WSR = 300
CLIP_T3 = 8.0  # symmetric clip => mean 0 preserved (documented relaxation)


def _gaussian_streams(
    rng: np.random.Generator, n_reps: int, t_len: int, mu: float = 0.0, sigma: float = 1.0
) -> Array:
    return np.asarray(mu + sigma * rng.standard_normal((n_reps, t_len)), dtype=float)


def _clipped_t3_streams(rng: np.random.Generator, n_reps: int, t_len: int) -> Array:
    return np.asarray(np.clip(rng.standard_t(3, (n_reps, t_len)), -CLIP_T3, CLIP_T3), dtype=float)


# ---------------------------------------------------------------------------
# (a) time-uniform coverage on SYNTHETIC Gaussian streams (sub-Gaussian CS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", ["two_sided_mixture", "poly_stitching", "poly_hedge_epsilon"])
def test_subgaussian_cs_time_uniform_coverage_gaussian(boundary: str) -> None:
    """P(any-time misscoverage) <= alpha across ALL sample times, N(0, 1)."""
    rng = np.random.default_rng(SEED)
    streams = _gaussian_streams(rng, N_REPS, T_GAUSS)
    cs = subgaussian_cs(streams, sigma=1.0, alpha=ALPHA, boundary=boundary)
    assert cs.lower.shape == (N_REPS, T_GAUSS)
    assert bool(np.all(np.isfinite(cs.lower))) and bool(np.all(np.isfinite(cs.upper)))
    assert bool(np.all(cs.lower < cs.upper))
    res = check_time_uniform_coverage(cs.lower, cs.upper, 0.0)
    assert float(res["violation_rate"]) <= ALPHA + VIOL_TOL
    per_time = np.asarray(res["per_time_coverage"], dtype=float)
    assert float(per_time.min()) >= 1.0 - ALPHA - PERTIME_TOL


def test_subgaussian_cs_covers_shifted_scaled_gaussian() -> None:
    """N(2.5, 0.7^2) with the matching declared proxy sigma: coverage holds."""
    rng = np.random.default_rng(SEED + 1)
    streams = _gaussian_streams(rng, 150, 800, mu=2.5, sigma=0.7)
    cs = subgaussian_cs(streams, sigma=0.7, alpha=ALPHA)
    res = check_time_uniform_coverage(cs.lower, cs.upper, 2.5)
    assert float(res["violation_rate"]) <= ALPHA + VIOL_TOL


# ---------------------------------------------------------------------------
# (b) time-uniform coverage on SYNTHETIC bounded streams (WSR + emp-Bernstein)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "mu"),
    [("uniform", 0.5), ("beta", 2.0 / 7.0)],
)
def test_wsr_cs_time_uniform_coverage_bounded(kind: str, mu: float) -> None:
    """Hedged-capital betting CS: any-time violation rate <= alpha + slack."""
    rng = np.random.default_rng(SEED + 2)
    if kind == "uniform":
        streams = rng.random((N_REPS_WSR, T_WSR))
    else:
        streams = rng.beta(2.0, 5.0, (N_REPS_WSR, T_WSR))
    cs = wsr_cs(streams, lower=0.0, upper=1.0, alpha=ALPHA, n_bisect=18)
    assert cs.lower.shape == (N_REPS_WSR, T_WSR)
    assert bool(np.all(cs.lower >= 0.0)) and bool(np.all(cs.upper <= 1.0))
    assert bool(np.all(cs.lower <= cs.upper))
    res = check_time_uniform_coverage(cs.lower, cs.upper, mu)
    assert float(res["violation_rate"]) <= ALPHA + VIOL_TOL


@pytest.mark.parametrize(
    ("kind", "mu"),
    [("uniform", 0.5), ("beta", 2.0 / 7.0)],
)
def test_empirical_bernstein_time_uniform_coverage_bounded(kind: str, mu: float) -> None:
    """Stitched empirical-Bernstein CS (Thm 4): any-time coverage control."""
    rng = np.random.default_rng(SEED + 3)
    if kind == "uniform":
        streams = rng.random((N_REPS, T_GAUSS))
    else:
        streams = rng.beta(2.0, 5.0, (N_REPS, T_GAUSS))
    cs = empirical_bernstein_cs(streams, lower=0.0, upper=1.0, alpha=ALPHA)
    assert bool(np.all(cs.lower >= 0.0)) and bool(np.all(cs.upper <= 1.0))
    res = check_time_uniform_coverage(cs.lower, cs.upper, mu)
    assert float(res["violation_rate"]) <= ALPHA + VIOL_TOL
    per_time = np.asarray(res["per_time_coverage"], dtype=float)
    assert float(per_time.min()) >= 1.0 - ALPHA - PERTIME_TOL


# ---------------------------------------------------------------------------
# (c) heavy tails: t(3) with the documented symmetric-clip relaxation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["wsr", "empirical_bernstein", "subgaussian"])
def test_heavy_tailed_clipped_t3_coverage(method: str) -> None:
    """t(df=3) clipped symmetrically to [-8, 8] (mean 0 preserved): covered.

    Unclipped t3 has no finite sub-Gaussian proxy variance and no bounded
    support, so all three constructions need the clip (module-docstring
    relaxation); sigma = 8 = (b - a)/2 is the Hoeffding proxy for the clip.
    """
    rng = np.random.default_rng(SEED + 4)
    streams = _clipped_t3_streams(rng, 100, 400)
    if method == "wsr":
        cs = wsr_cs(streams, lower=-CLIP_T3, upper=CLIP_T3, alpha=ALPHA, n_bisect=16)
    elif method == "empirical_bernstein":
        cs = empirical_bernstein_cs(streams, lower=-CLIP_T3, upper=CLIP_T3, alpha=ALPHA)
    else:
        cs = subgaussian_cs(streams, sigma=CLIP_T3, alpha=ALPHA)
    res = check_time_uniform_coverage(cs.lower, cs.upper, 0.0)
    assert float(res["violation_rate"]) <= ALPHA + VIOL_TOL


# ---------------------------------------------------------------------------
# (d) width shrinks ~ 1/sqrt(t)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", ["two_sided_mixture", "poly_stitching", "poly_hedge_epsilon"])
def test_subgaussian_width_shrinks_inverse_sqrt(boundary: str) -> None:
    """Log-log width slope ~ -0.5 and the 1000 -> 4000 width ratio ~ 1/2."""
    rng = np.random.default_rng(SEED + 5)
    stream = rng.standard_normal(4000)
    cs = subgaussian_cs(stream, sigma=1.0, alpha=ALPHA, boundary=boundary)
    slope = width_decay_slope(cs.lower, cs.upper)
    assert -0.62 <= slope <= -0.38  # 1/sqrt(t) up to the loglog factor
    widths = cs_widths(cs.lower, cs.upper)
    ratio = float(widths[3999] / widths[999])
    assert 0.40 <= ratio <= 0.62  # ideal sqrt(1000/4000) = 0.5


def test_wsr_width_shrinks_over_time() -> None:
    """Betting CS width decays; early decay is faster than 1/sqrt(t) because
    the CS starts at (nearly) full support and the variance estimate adapts."""
    rng = np.random.default_rng(SEED + 6)
    stream = rng.beta(50.0, 50.0, 250)
    cs = wsr_cs(stream, lower=0.0, upper=1.0, alpha=ALPHA, n_bisect=20)
    slope = width_decay_slope(cs.lower, cs.upper, from_frac=0.25)
    assert -1.2 <= slope <= -0.3
    widths = cs_widths(cs.lower, cs.upper)
    assert float(widths[-1]) < float(widths[widths.size // 4])


def test_width_ratio_vs_fixed_within_small_factor() -> None:
    """Price of anytime validity: ~1-2x the fixed-sample CI width at T=4000
    (Howard et al. 2021 report the mixture stays within a factor ~2)."""
    rng = np.random.default_rng(SEED + 7)
    stream = rng.standard_normal(4000)
    cs = subgaussian_cs(stream, sigma=1.0, alpha=ALPHA)
    ratio = width_ratio_vs_fixed(cs.lower, cs.upper, stream, ALPHA, sigma=1.0)
    assert 1.0 < ratio <= 2.5


# ---------------------------------------------------------------------------
# (e) variance adaptivity: empirical-Bernstein / WSR beat the Hoeffding proxy
# ---------------------------------------------------------------------------


def test_empirical_bernstein_adapts_to_low_variance() -> None:
    """beta(50, 50) (var ~ 0.0025) in [0, 1]: EB width << sigma = 0.5 CS."""
    rng = np.random.default_rng(SEED + 8)
    stream = rng.beta(50.0, 50.0, 1000)
    eb = empirical_bernstein_cs(stream, lower=0.0, upper=1.0, alpha=ALPHA)
    mix = subgaussian_cs(stream, sigma=0.5, alpha=ALPHA)  # Hoeffding proxy
    eb_w = float(cs_widths(eb.lower, eb.upper)[-1])
    mix_w = float(cs_widths(mix.lower, mix.upper)[-1])
    assert eb_w <= 0.6 * mix_w


def test_wsr_beats_hoeffding_width_on_low_variance_data() -> None:
    """Betting adapts to the unknown small variance; the fixed proxy does not."""
    rng = np.random.default_rng(SEED + 9)
    stream = rng.beta(50.0, 50.0, 250)
    w = wsr_cs(stream, lower=0.0, upper=1.0, alpha=ALPHA, n_bisect=20)
    mix = subgaussian_cs(stream, sigma=0.5, alpha=ALPHA)
    w_w = float(cs_widths(w.lower, w.upper)[-1])
    mix_w = float(cs_widths(mix.lower, mix.upper)[-1])
    assert w_w <= 0.6 * mix_w


# ---------------------------------------------------------------------------
# boundary math: closed forms and constants pinned to the papers
# ---------------------------------------------------------------------------


def test_normal_mixture_boundary_closed_form() -> None:
    """u(v) = sqrt((v+rho) log((v+rho)/(alpha^2 rho))) (Howard et al. Eq. 14)."""
    v, a, rho = 100.0, 0.05, 10.0
    u = normal_mixture_boundary(np.array([v]), a, rho)
    assert float(u[0]) == pytest.approx(30.37811021425952, rel=1e-12)
    # u(0) = rho-scaled intercept; boundary strictly increasing in v.
    u0 = normal_mixture_boundary(np.array([0.0]), a, rho)
    assert float(u0[0]) == pytest.approx(np.sqrt(rho * np.log(1.0 / a**2)), rel=1e-12)
    grid = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    u_all = normal_mixture_boundary(grid, a, rho)
    assert bool(np.all(np.diff(u_all) > 0.0))


def test_optimal_rho_minimizes_boundary_at_target_time() -> None:
    """Proposition 3(b): rho* minimizes the mixture boundary at v_opt."""
    v_opt, a = 1000.0, 0.05
    rho_star = optimal_mixture_rho(v_opt, a)
    u_star = float(normal_mixture_boundary(np.array([v_opt]), a, rho_star)[0])
    u_half = float(normal_mixture_boundary(np.array([v_opt]), a, rho_star / 2.0)[0])
    u_double = float(normal_mixture_boundary(np.array([v_opt]), a, 2.0 * rho_star)[0])
    assert u_star <= u_half
    assert u_star <= u_double


def test_poly_stitching_matches_howard_equation_11() -> None:
    """Exact stitching boundary vs the paper's rounded constants (Eq. 11):
    1.7*sqrt(t*(log log(2t) + 0.72*log(5.2/alpha))) at eta=2, s=1.4, m=1."""
    v = np.array([1000.0])
    u = float(poly_stitching_boundary(v, ALPHA, c=0.0, eta=2.0, s=1.4, m=1.0)[0])
    paper = 1.7 * np.sqrt(1000.0 * (np.log(np.log(2000.0)) + 0.72 * np.log(5.2 / ALPHA)))
    assert u == pytest.approx(paper, rel=0.05)
    assert u == pytest.approx(124.16547101588009, rel=1e-9)  # pinned for determinism


def test_hedge_boundaries_positive_monotone_and_alpha_ordered() -> None:
    """Hedge-epsilon boundaries: positive, increasing in v, wider for smaller
    alpha; the polynomial-budget variant is increasing in v as well."""
    grid = np.array([1.0, 5.0, 25.0, 125.0, 625.0])
    u = hedge_epsilon_boundary(grid, ALPHA, rho=1.0, epsilon=0.5)
    assert bool(np.all(u > 0.0))
    assert bool(np.all(np.diff(u) > 0.0))
    u_tight = hedge_epsilon_boundary(grid, 0.01, rho=1.0, epsilon=0.5)
    u_loose = hedge_epsilon_boundary(grid, 0.10, rho=1.0, epsilon=0.5)
    assert bool(np.all(u_tight > u)) and bool(np.all(u > u_loose))
    p = poly_hedge_epsilon_boundary(grid, ALPHA, rho=1.0, epsilon=0.5, eta=2.0, s=1.4)
    assert bool(np.all(p > 0.0))
    assert bool(np.all(np.diff(p) > 0.0))


# ---------------------------------------------------------------------------
# mechanics: shapes, determinism, coverage helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda z: subgaussian_cs(z, sigma=1.0, alpha=ALPHA),
        lambda z: subgaussian_cs(z, sigma=1.0, alpha=ALPHA, boundary="poly_stitching"),
        lambda z: empirical_bernstein_cs(z, lower=-4.0, upper=4.0, alpha=ALPHA),
        lambda z: wsr_cs(np.clip(z, -4.0, 4.0), lower=-4.0, upper=4.0, alpha=ALPHA, n_bisect=12),
    ],
    ids=["mixture", "stitching", "empirical-bernstein", "wsr"],
)
def test_cs_shapes_and_determinism(build) -> None:
    """1-D -> 1-D, 2-D -> 2-D; repeated calls are bit-identical (no RNG)."""
    rng = np.random.default_rng(SEED + 10)
    stream1d = rng.standard_normal(60)
    stream2d = rng.standard_normal((5, 60))
    cs1a, cs1b = build(stream1d), build(stream1d)
    cs2a, cs2b = build(stream2d), build(stream2d)
    assert isinstance(cs1a, ConfidenceSequence)
    assert cs1a.lower.shape == (60,) and cs1a.upper.shape == (60,)
    assert cs2a.lower.shape == (5, 60) and cs2a.upper.shape == (5, 60)
    assert cs1a.n == 60 and cs1a.alpha == ALPHA
    np.testing.assert_array_equal(cs1a.lower, cs1b.lower)
    np.testing.assert_array_equal(cs1a.upper, cs1b.upper)
    np.testing.assert_array_equal(cs2a.lower, cs2b.lower)
    np.testing.assert_array_equal(cs2a.upper, cs2b.upper)
    np.testing.assert_allclose(cs1a.widths, cs1a.upper - cs1a.lower, rtol=0, atol=0)


def test_wsr_mc_reproducible_from_seed() -> None:
    """Same seed -> identical streams -> identical violation counts (pinned)."""
    results = []
    for _ in range(2):
        rng = np.random.default_rng(SEED + 11)
        streams = rng.random((20, 120))
        cs = wsr_cs(streams, lower=0.0, upper=1.0, alpha=ALPHA, n_bisect=14)
        results.append(check_time_uniform_coverage(cs.lower, cs.upper, 0.5))
    np.testing.assert_array_equal(
        results[0]["first_violations"],
        results[1]["first_violations"],  # type: ignore[arg-type]
    )
    assert results[0]["violation_rate"] == results[1]["violation_rate"]


def test_check_time_uniform_coverage_mechanics() -> None:
    lo = np.zeros(10)
    up = np.zeros(10)
    mu = np.zeros(10)
    mu[7] = 0.5  # planted miss at t = 7 (0-based)
    res = check_time_uniform_coverage(lo, up, mu)
    assert res["n_reps"] == 1
    assert float(res["violation_rate"]) == 1.0
    assert float(res["time_uniform_coverage"]) == 0.0
    assert np.asarray(res["first_violations"]).tolist() == [7]
    per_time = np.asarray(res["per_time_coverage"], dtype=float)
    assert per_time.tolist() == [1.0] * 7 + [0.0] + [1.0] * 2
    # Scalar mu, batch input: half the rows violated; -1 sentinel otherwise.
    lo2 = np.tile(lo, (4, 1))
    up2 = np.tile(up, (4, 1))
    mu2 = np.zeros((4, 10))
    mu2[1, 3] = 1.0
    mu2[2, 9] = 1.0
    res2 = check_time_uniform_coverage(lo2, up2, mu2)
    assert res2["n_reps"] == 4
    assert float(res2["time_uniform_coverage"]) == 0.5
    assert np.asarray(res2["first_violations"]).tolist() == [-1, 3, 9, -1]
    # Scalar mu broadcasting and infinite endpoints.
    res3 = check_time_uniform_coverage(lo, up, 0.0)
    assert float(res3["violation_rate"]) == 0.0
    res4 = check_time_uniform_coverage(np.full(5, -np.inf), np.full(5, np.inf), 3.0)
    assert float(res4["time_uniform_coverage"]) == 1.0


def test_fixed_sample_ci_width_closed_form() -> None:
    x = np.arange(1.0, 101.0)
    w = fixed_sample_ci_width(x, ALPHA, sigma=2.0)
    from scipy.stats import norm

    assert w == pytest.approx(2.0 * float(norm.ppf(0.975)) * 2.0 / 10.0, rel=1e-12)
    # Without sigma: sample sd (ddof=1).
    w2 = fixed_sample_ci_width(x, ALPHA)
    assert w2 == pytest.approx(2.0 * float(norm.ppf(0.975)) * float(np.std(x, ddof=1)) / 10.0)


def test_width_decay_slope_exact_inverse_sqrt() -> None:
    t = np.arange(1, 501, dtype=float)
    half = 3.0 / np.sqrt(t)
    assert width_decay_slope(-half, half) == pytest.approx(-0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# (f) fail-closed edges
# ---------------------------------------------------------------------------

_TINY = np.array([0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: subgaussian_cs([], sigma=1.0),
        lambda: subgaussian_cs([np.nan], sigma=1.0),
        lambda: subgaussian_cs([np.inf], sigma=1.0),
        lambda: subgaussian_cs(np.zeros((2, 2, 2)), sigma=1.0),  # 3-D rejected
        lambda: subgaussian_cs(_TINY, sigma=0.0),
        lambda: subgaussian_cs(_TINY, sigma=-1.0),
        lambda: subgaussian_cs(_TINY, sigma=np.nan),
        lambda: subgaussian_cs(_TINY, sigma=1.0, alpha=0.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, alpha=1.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, alpha=np.nan),
        lambda: subgaussian_cs(_TINY, sigma=1.0, alpha=-0.1),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="bogus"),
        lambda: subgaussian_cs(_TINY, sigma=1.0, rho=0.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, rho=-1.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="poly_stitching", m=0.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="poly_stitching", eta=1.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="poly_stitching", s=1.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="poly_hedge_epsilon", epsilon=0.0),
        lambda: subgaussian_cs(_TINY, sigma=1.0, boundary="poly_hedge_epsilon", epsilon=-1.0),
        lambda: wsr_cs([0.1, 1.5], lower=0.0, upper=1.0),  # outside support
        lambda: wsr_cs([0.1, -0.5], lower=0.0, upper=1.0),
        lambda: wsr_cs(_TINY, lower=1.0, upper=0.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=0.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=np.inf),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, theta=0.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, theta=1.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, truncation=0.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, truncation=1.0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, n_bisect=0),
        lambda: wsr_cs(_TINY, lower=0.0, upper=1.0, alpha=1.5),
        lambda: wsr_cs(np.zeros(5000), lower=0.0, upper=1.0),  # exceeds _WSR_MAX_T
        lambda: empirical_bernstein_cs([0.1, 2.0], lower=0.0, upper=1.0),
        lambda: empirical_bernstein_cs([], lower=0.0, upper=1.0),
        lambda: empirical_bernstein_cs(_TINY, lower=0.0, upper=1.0, m=-1.0),
        lambda: empirical_bernstein_cs(_TINY, lower=0.0, upper=1.0, eta=0.5),
        lambda: empirical_bernstein_cs(_TINY, lower=0.0, upper=1.0, s=0.9),
    ],
)
def test_fail_closed_builders_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: normal_mixture_boundary([-1.0], 0.05, 1.0),
        lambda: normal_mixture_boundary([np.nan], 0.05, 1.0),
        lambda: normal_mixture_boundary([], 0.05, 1.0),
        lambda: normal_mixture_boundary([1.0], 0.05, 0.0),
        lambda: normal_mixture_boundary([1.0], 0.0, 1.0),
        lambda: normal_mixture_boundary([1.0], 1.0, 1.0),
        lambda: poly_stitching_boundary([1.0], 0.05, c=-1.0),
        lambda: poly_stitching_boundary([1.0], 0.05, eta=1.0),
        lambda: poly_stitching_boundary([1.0], 0.05, s=0.9),
        lambda: poly_stitching_boundary([1.0], 0.05, m=0.0),
        lambda: poly_stitching_boundary([-1.0], 0.05),
        lambda: hedge_epsilon_boundary([1.0], 0.05, rho=0.0),
        lambda: hedge_epsilon_boundary([1.0], 0.05, rho=1.0, epsilon=0.0),
        lambda: hedge_epsilon_boundary([1.0], 0.05, rho=1.0, epsilon=np.nan),
        lambda: poly_hedge_epsilon_boundary([-2.0], 0.05, rho=1.0),
        lambda: poly_hedge_epsilon_boundary([1.0], 0.05, rho=1.0, eta=1.0),
        lambda: poly_hedge_epsilon_boundary([1.0], 0.05, rho=0.0),
        lambda: optimal_mixture_rho(0.0, 0.05),
        lambda: optimal_mixture_rho(100.0, 1.0),
        lambda: optimal_mixture_rho(-1.0, 0.05),
    ],
)
def test_fail_closed_boundaries_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: check_time_uniform_coverage([0.0, 0.0], [1.0], 0.5),  # shape mismatch
        lambda: check_time_uniform_coverage([np.nan], [1.0], 0.5),
        lambda: check_time_uniform_coverage([0.0], [np.nan], 0.5),
        lambda: check_time_uniform_coverage([0.0], [1.0], np.nan),
        lambda: check_time_uniform_coverage([0.0, 0.0], [1.0, 1.0], [0.5, 0.5, 0.5]),
        lambda: check_time_uniform_coverage([], [], 0.0),
        lambda: width_decay_slope(np.zeros((2, 10)), np.ones((2, 10))),  # 2-D rejected
        lambda: width_decay_slope([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], from_frac=1.0),
        lambda: width_decay_slope([0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),  # zero width in window
        lambda: fixed_sample_ci_width(np.zeros((2, 5)), 0.05),  # 2-D rejected
        lambda: fixed_sample_ci_width([1.0, 1.0, 1.0], 0.05),  # zero sample sd
        lambda: fixed_sample_ci_width([1.0, 2.0, 3.0], 0.05, sigma=0.0),
        lambda: fixed_sample_ci_width([1.0, 2.0, 3.0], 0.0),
        lambda: width_ratio_vs_fixed(
            np.zeros((2, 5)), np.ones((2, 5)), [1.0, 2.0, 3.0, 4.0, 5.0], 0.05
        ),
        lambda: cs_widths([0.0, 0.0], [1.0]),
        lambda: cs_widths([], []),
    ],
)
def test_fail_closed_helpers_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()
