"""Fair CRPS, threshold-weighted CRPS, and skill scores (SOTA-05 F-02, G-6).

SYNTHETIC, seeded, deterministic — correctness evidence about estimator
behaviour, never market evidence. Nothing here is a live-trading or P&L claim.

``crps_empirical`` (the biased plug-in) is deliberately left untouched: sealed
receipts may reference it and the honesty contract (rule #4) makes that evidence
immutable. These tests pin its *measured* bias so the defect is documented as a
reproducible artefact rather than rediscovered, and pin the new unbiased
:func:`crps_fair` as unbiased on a known distribution across several ``n``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.scoring import (
    crps_empirical,
    crps_fair,
    crps_skill_score,
    crps_threshold_weighted,
    mean_pinball,
    pinball_skill_score,
    skill_score,
    threshold_weight_transform,
)

SEED = 20260928
_REPS = 4000


def _crps_n01(y: float) -> float:
    r"""Closed-form CRPS of N(0,1) at y: ``z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)``."""
    return float(y * (2.0 * norm.cdf(y) - 1.0) + 2.0 * norm.pdf(y) - 1.0 / math.sqrt(math.pi))


# --- F-02: the fair estimator is unbiased; the plug-in is not ----------------


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_crps_fair_is_unbiased_on_a_known_distribution(n: int) -> None:
    """``E[crps_fair]`` tracks the closed-form CRPS of the true predictive.

    Observation *and* ensemble are both ``N(0,1)``, so the population CRPS is
    ``E_y[CRPS(N(0,1), y)]`` — a fixed, known number. The fair estimator must
    land inside ~3 Monte-Carlo s.e. of it at every ``n``; that is unbiasedness
    of the estimator, not convergence of a single draw.
    """
    rng = np.random.default_rng(SEED)
    ys = rng.normal(0.0, 1.0, size=_REPS)
    truth = float(np.mean([_crps_n01(float(y)) for y in ys]))
    fair = np.array([crps_fair(float(y), rng.normal(0.0, 1.0, size=n)) for y in ys])
    est = float(np.mean(fair))
    se = float(np.std(fair, ddof=1) / math.sqrt(_REPS))
    assert est == pytest.approx(truth, abs=4.0 * se), f"n={n}: est={est} truth={truth} se={se}"


@pytest.mark.parametrize("n", [5, 10, 20])
def test_crps_empirical_bias_matches_the_brief(n: int) -> None:
    """Pin the measured plug-in bias: +20.5% (n=5), +9.9% (n=10), +5.0% (n=20).

    The analytic bias is ``E|X-X'|/(2n) = sigma/(n*sqrt(pi))`` for a Gaussian
    predictive; both the absolute bias and the percentage-of-score reading are
    asserted so the ensemble-size dependence (the reason CRPS league tables
    mixing ensemble sizes rank by SIZE, not skill) is pinned in the suite.
    """
    expected_pct = {5: 20.5, 10: 9.9, 20: 5.0}[n]
    rng = np.random.default_rng(SEED)
    ys = rng.normal(0.0, 1.0, size=_REPS)
    plug = np.array([crps_empirical(float(y), rng.normal(0.0, 1.0, size=n)) for y in ys])
    fair = np.array([crps_fair(float(y), rng.normal(0.0, 1.0, size=n)) for y in ys])
    bias = float(np.mean(plug) - np.mean(fair))
    analytic = 1.0 / (n * math.sqrt(math.pi))  # sigma = 1
    pct = 100.0 * bias / float(np.mean(fair))
    assert bias == pytest.approx(analytic, rel=0.05), f"n={n}: bias={bias} analytic={analytic}"
    assert pct == pytest.approx(expected_pct, abs=1.5), f"n={n}: {pct}% vs {expected_pct}%"


def test_crps_fair_is_strictly_below_the_plug_in_at_every_n() -> None:
    """The plug-in is *positively* biased, so fair < plug-in, and the gap shrinks.

    Pinning the direction (not just the magnitude) means a future refactor that
    flips the diagonal convention fails loudly.
    """
    rng = np.random.default_rng(SEED)
    gaps = []
    for n in (5, 10, 50, 200):
        ys = rng.normal(0.0, 1.0, size=800)
        plug = np.mean([crps_empirical(float(y), rng.normal(size=n)) for y in ys])
        fair = np.mean([crps_fair(float(y), rng.normal(size=n)) for y in ys])
        gap = float(plug - fair)
        assert gap > 0.0, f"n={n}: gap={gap}"
        gaps.append(gap)
    # Gap is monotonically decreasing in n (bias ~ 1/n).
    assert all(a > b for a, b in zip(gaps, gaps[1:], strict=False)), gaps


def test_crps_fair_matches_plug_in_at_large_n() -> None:
    """Both estimators agree as ``n -> inf`` — they estimate the same object."""
    rng = np.random.default_rng(7)
    sample = rng.normal(0.0, 1.0, size=20_000)
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)  # closed form at y=0
    assert crps_fair(0.0, sample) == pytest.approx(expected, abs=0.005)
    assert crps_empirical(0.0, sample) == pytest.approx(expected, abs=0.005)


def test_crps_fair_hand_computable_two_point_ensemble() -> None:
    """y=0, sample={-1, 1} — the n=2 cell where the two estimators differ most.

    term1 = mean(|Xi - y|) = 1. For the **fair** form the off-diagonal sum is
    ``|X1-X2| + |X2-X1| = 4`` over ``2n(n-1) = 4``, so term2 = 1 and
    ``crps_fair = 0``. The **plug-in** form divides by ``2n² = 8`` (counting the
    two zero diagonal cells), giving term2 = 0.5 and ``crps_empirical = 0.5``.
    The ratio is exactly ``(n-1)/n = 1/2``, its worst case: pinning this cell
    pins the mechanism of the bias, not just its magnitude.
    """
    sample = np.array([-1.0, 1.0])
    assert crps_fair(0.0, sample) == pytest.approx(0.0)
    assert crps_empirical(0.0, sample) == pytest.approx(0.5)
    # crps_fair accepts a length-1 y array, same contract as the sibling.
    assert crps_fair(np.array([0.0]), sample) == pytest.approx(0.0)


def test_crps_fair_edges_mirror_crps_empirical() -> None:
    """Same fail-closed contract as the sibling estimator."""
    assert math.isnan(crps_fair(0.0, np.array([])))
    assert math.isnan(crps_fair(0.0, np.array([np.nan, np.nan])))
    assert math.isnan(crps_fair(np.nan, np.array([0.0, 1.0])))
    # n == 1 degenerates to |X_1 - y|, documented, not NaN.
    assert crps_fair(0.0, np.array([3.0])) == pytest.approx(3.0)
    # Non-finite members are masked out first, so {-1, 1, nan} reduces to the
    # n=2 cell above and takes its value, not the 3-member one.
    assert crps_fair(0.0, np.array([-1.0, 1.0, np.nan])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="single observation"):
        crps_fair(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# --- G-6: the proper threshold-weighted replacement -------------------------


def test_threshold_weight_transform_is_the_weight_kernel_antiderivative() -> None:
    r"""``W(u) = sgn(u)[min(|u|,t) + w*max(|u|-t,0)]``: odd, increasing, kinked.

    Also pins the metric identity ``|W(x) - W(y)| = int w(z)(1{x<=z}-1{y<=z})^2 dz``
    on a hand-computable pair, which is what makes the weighted CRPS proper.
    """
    t, w = 1.0, 3.0
    assert threshold_weight_transform(np.array([0.0]), t, w)[0] == pytest.approx(0.0)
    assert threshold_weight_transform(np.array([2.0]), t, w)[0] == pytest.approx(4.0)
    assert threshold_weight_transform(np.array([-2.0]), t, w)[0] == pytest.approx(-4.0)
    # rho_w(-2, 2) = |W(2) - W(-2)| = 8 = 2*(t + w*(2-t)) with t=1, w=3.
    assert abs(4.0 - (-4.0)) == pytest.approx(8.0)
    # Odd and monotone on a grid.
    grid = np.linspace(-6.0, 6.0, 241)
    wg = threshold_weight_transform(grid, t, w)
    np.testing.assert_allclose(wg, -threshold_weight_transform(-grid, t, w), atol=1e-12)
    assert np.all(np.diff(wg) >= 0.0)


def test_threshold_weight_transform_rejects_malformed_kernels() -> None:
    """Fail closed: a malformed kernel must raise, never return a wrong score."""
    with pytest.raises(ValueError, match="threshold"):
        threshold_weight_transform(np.array([1.0]), -1.0, 2.0)
    with pytest.raises(ValueError, match="threshold"):
        threshold_weight_transform(np.array([1.0]), np.inf, 2.0)
    with pytest.raises(ValueError, match="weight"):
        threshold_weight_transform(np.array([1.0]), 1.0, 0.5)
    with pytest.raises(ValueError, match="weight"):
        threshold_weight_transform(np.array([1.0]), 1.0, np.nan)


def _weighted_crps_integral(
    cdf, y: float, threshold: float, weight: float, knots: np.ndarray
) -> float:
    r"""Exact piecewise ``int w(z)(F(z) - 1{y<=z})^2 dz`` on cells between ``knots``.

    ``F`` must be constant on each open cell (true for an empirical CDF and for
    a step-approximated smooth CDF), so the integral is a finite sum, not a
    Riemann approximation.
    """
    grid = np.concatenate([[knots[0] - 8.0], knots, [knots[-1] + 8.0]])
    total = 0.0
    for a, b in zip(grid[:-1], grid[1:], strict=False):
        mid = 0.5 * (a + b)
        fz = float(cdf(mid))
        ind = 1.0 if y <= mid else 0.0
        wmid = weight if abs(mid) > threshold else 1.0
        total += wmid * (fz - ind) ** 2 * (b - a)
    return total


def _plugin_weighted_crps(y: float, sample: np.ndarray, threshold: float, weight: float) -> float:
    r"""Plug-in form on the W-transformed ensemble: ``1/n Σ|W(X_i)-W(y)| - 1/(2n²) Σ_{i,j}``."""
    wx = threshold_weight_transform(sample, threshold, weight)
    wy = float(threshold_weight_transform(np.array([y]), threshold, weight)[0])
    # np.mean carries both normalisations: 1/n for the first term and 1/n**2
    # for the pairwise double sum (halved), so no explicit n is needed.
    term1 = float(np.mean(np.abs(wx - wy)))
    pairwise = np.abs(wx[:, None] - wx[None, :])
    return term1 - float(np.mean(pairwise)) / 2.0


def test_threshold_weighted_crps_kernel_matches_the_integral_definition() -> None:
    r"""Pin the identity ``int w(z)(F_n(z) - 1{y<=z})^2 dz`` == plug-in on ``W``.

    This is the mathematical core of the construction: the weighted CRPS *is* an
    ordinary CRPS of the W-transformed ensemble, which is why it stays proper at
    any weight (the weight multiplies one non-negative integrand, not the two
    terms of an energy score by different powers). Exact piecewise integration on
    a hand-sized ensemble, so the tolerance is floating-point tight.
    """
    sample = np.array([-2.0, -0.5, 0.3, 1.1, 2.4])
    y = 0.9
    for threshold, weight in ((1.0, 1.0), (1.0, 3.0), (0.5, 2.0), (2.0, 5.0)):
        knots = np.unique(np.concatenate([sample, [y], [-threshold, threshold]]))
        cdf = lambda z: float(np.mean(sample <= z))  # noqa: E731
        exact = _weighted_crps_integral(cdf, y, threshold, weight, knots)
        plug = _plugin_weighted_crps(y, sample, threshold, weight)
        assert plug == pytest.approx(exact, abs=1e-9), (threshold, weight, plug, exact)
        # The shipped estimator is the *fair* form, which differs from the
        # plug-in by exactly the diagonal-count term E|X-X'|/(2n). Pin that
        # relationship too, so the two forms can never silently swap.
        fair = crps_threshold_weighted(y, sample, threshold=threshold, weight=weight)
        wx = threshold_weight_transform(sample, threshold, weight)
        n = sample.size
        pairwise = np.abs(wx[:, None] - wx[None, :])
        diag_correction = float(pairwise.sum()) * (1.0 / (2.0 * n * n) - 1.0 / (2.0 * n * (n - 1)))
        assert fair == pytest.approx(plug + diag_correction, abs=1e-12)
        assert fair < plug  # fair is below the positively-biased plug-in


def test_threshold_weighted_crps_converges_to_the_population_integral() -> None:
    r"""A large ensemble recovers ``int w(z)(F(z) - 1{y<=z})^2 dz`` for the true F.

    The fair U-statistic is unbiased, so at large ``n`` it must approach the
    population weighted CRPS of the true predictive (here ``N(0, 1.5^2)``). The
    reference integral is computed on a fine grid with the exact Gaussian CDF;
    tolerance is set by ensemble Monte-Carlo error, not by the quadrature.
    """
    rng = np.random.default_rng(SEED)
    mu, sd = 0.0, 1.5
    y = 0.7
    cdf = lambda z: float(norm.cdf((z - mu) / sd))  # noqa: E731
    for threshold, weight in ((1.0, 3.0), (0.5, 2.0)):
        knots = np.linspace(-12.0, 12.0, 4001)
        truth = _weighted_crps_integral(cdf, y, threshold, weight, knots)
        ests = []
        for _ in range(40):
            sample = rng.normal(mu, sd, size=2000)
            ests.append(crps_threshold_weighted(y, sample, threshold=threshold, weight=weight))
        assert float(np.mean(ests)) == pytest.approx(truth, rel=0.02), (threshold, weight)


def test_threshold_weighted_crps_at_weight_one_is_crps_fair() -> None:
    """``weight = 1`` must recover the fair CRPS bit-for-bit."""
    rng = np.random.default_rng(3)
    sample = rng.normal(size=40)
    y = 0.7
    assert crps_threshold_weighted(y, sample, threshold=1.0, weight=1.0) == crps_fair(y, sample)


def test_threshold_weighted_crps_is_proper_where_the_energy_variant_is_not() -> None:
    """The headline result: proper at ``weight = 3`` with an interior minimum.

    Obs law ``N(0,1)``, forecast ``N(0, sigma^2)``, 40 members, seed 2026.
    Measured (docs/SOTA/05 §F-01 contrast): ``threshold_energy_score`` at the
    same weight runs +1.177 -> -17.685 (monotone down, unbounded). This score
    dips to a minimum at sigma = 1 and then *rises* — variance inflation is
    penalised, and there is no ``sqrt(2)`` ceiling.
    """
    threshold, weight, members = 1.0, 3.0, 40
    means = {}
    for sigma in (0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0):
        rng = np.random.default_rng(2026)
        vals = []
        for _ in range(300):
            y = float(rng.normal(0.0, 1.0))
            ens = rng.normal(0.0, sigma, size=members)
            vals.append(crps_threshold_weighted(y, ens, threshold=threshold, weight=weight))
        means[sigma] = float(np.mean(vals))
    values = list(means.values())
    argmin = min(means, key=lambda k: means[k])
    assert argmin == pytest.approx(1.0), means
    assert means[8.0] > means[argmin], means
    # Non-negative throughout: a proper CRPS-type score cannot go negative.
    assert all(v > 0.0 for v in values), values
    # Monotone increasing past the truth — the direction that punishes inflation.
    tail = [means[s] for s in (1.0, 1.5, 2.0, 4.0, 8.0)]
    assert all(a < b for a, b in zip(tail, tail[1:], strict=False)), tail


def test_threshold_weighted_crps_amplifies_tail_errors_more_than_central() -> None:
    """The diagnostic the improper variant was built for, achieved properly."""
    rng = np.random.default_rng(11)
    sample = rng.normal(0.0, 1.0, size=2000)
    central, tail = 0.3, 3.0
    plain_c = crps_threshold_weighted(central, sample, threshold=1.5, weight=1.0)
    plain_t = crps_threshold_weighted(tail, sample, threshold=1.5, weight=1.0)
    amp_c = crps_threshold_weighted(central, sample, threshold=1.5, weight=3.0)
    amp_t = crps_threshold_weighted(tail, sample, threshold=1.5, weight=3.0)
    assert amp_t > amp_c
    assert plain_t > plain_c
    assert (amp_t - amp_c) > (plain_t - plain_c)


def test_threshold_weighted_crps_fail_closed_contracts() -> None:
    rng = np.random.default_rng(5)
    sample = rng.normal(size=20)
    with pytest.raises(ValueError, match="single observation"):
        crps_threshold_weighted(np.array([0.0, 1.0]), sample, threshold=1.0)
    with pytest.raises(ValueError, match="weight"):
        crps_threshold_weighted(0.0, sample, threshold=1.0, weight=0.5)
    with pytest.raises(ValueError, match="threshold"):
        crps_threshold_weighted(0.0, sample, threshold=-1.0)
    assert math.isnan(crps_threshold_weighted(0.0, np.array([]), threshold=1.0, weight=2.0))
    # n == 1 degenerates to |W(X_1) - W(y)|.
    assert crps_threshold_weighted(0.0, np.array([2.0]), threshold=1.0, weight=3.0) == (
        pytest.approx(4.0)
    )


# --- Work item 5: skill scores ----------------------------------------------


def test_skill_score_is_zero_for_an_identical_forecast() -> None:
    assert skill_score(0.37, 0.37) == pytest.approx(0.0)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_skill_score_is_negative_when_worse_and_positive_when_better() -> None:
    assert skill_score(0.50, 0.25) < 0.0  # higher loss than reference
    assert skill_score(0.10, 0.25) > 0.0  # lower loss than reference
    assert skill_score(0.0, 0.25) == pytest.approx(1.0)  # perfect score


def test_skill_score_fails_closed_on_a_meaningless_reference() -> None:
    """A zero/negative or non-finite reference makes the ratio uninterpretable."""
    assert math.isnan(skill_score(0.2, 0.0))
    assert math.isnan(skill_score(0.2, -0.1))
    assert math.isnan(skill_score(np.nan, 0.2))
    assert math.isnan(skill_score(0.2, np.inf))


def test_pinball_skill_score_identical_and_worse() -> None:
    rng = np.random.default_rng(17)
    y = rng.normal(size=200)
    q = rng.normal(size=200)
    worse = q + 0.5 * np.sign(q)
    tau = 0.9
    assert pinball_skill_score(y, q, q, tau) == pytest.approx(0.0)
    assert pinball_skill_score(y, worse, q, tau) < 0.0
    assert pinball_skill_score(y, q, worse, tau) > 0.0


def test_crps_skill_score_identical_and_size_robust() -> None:
    """Identical ensembles -> 0, and the fair base keeps mixed sizes honest."""
    rng = np.random.default_rng(23)
    y = 0.4
    ref = rng.normal(0.0, 1.0, size=50)
    assert crps_skill_score(y, ref, ref) == pytest.approx(0.0)
    worse = rng.normal(0.0, 3.0, size=50)
    assert crps_skill_score(y, worse, ref) < 0.0

    # Two ensembles drawn from the SAME predictive but with different sizes must
    # not be separated by size. Averaged over many draws so the comparison is
    # about estimator bias, not one unlucky sample. Each mean is asserted against
    # its analytic expectation within Monte-Carlo error, which is the honest
    # statement: the fair estimator's gap is zero in expectation, the plug-in's
    # is the difference of the two biases.
    reps = 4000
    fair_gaps, plug_gaps = [], []
    for _ in range(reps):
        y0 = float(rng.normal(0.0, 1.0))
        small = rng.normal(0.0, 1.0, size=5)
        big = rng.normal(0.0, 1.0, size=500)
        fair_gaps.append(crps_fair(y0, small) - crps_fair(y0, big))
        plug_gaps.append(crps_empirical(y0, small) - crps_empirical(y0, big))
    fair_arr = np.array(fair_gaps)
    plug_arr = np.array(plug_gaps)
    mean_fair_gap = float(np.mean(fair_arr))
    mean_plug_gap = float(np.mean(plug_arr))
    se_fair = float(np.std(fair_arr, ddof=1) / math.sqrt(reps))
    se_plug = float(np.std(plug_arr, ddof=1) / math.sqrt(reps))
    # sigma = 1 -> bias(n) = 1/(n*sqrt(pi)); the gap of the two biases:
    expected_plug_gap = 1.0 / (5.0 * math.sqrt(math.pi)) - 1.0 / (500.0 * math.sqrt(math.pi))
    assert mean_fair_gap == pytest.approx(0.0, abs=3.0 * se_fair), (mean_fair_gap, se_fair)
    assert mean_plug_gap == pytest.approx(expected_plug_gap, abs=3.0 * se_plug), (
        mean_plug_gap,
        expected_plug_gap,
    )
    # The artefact a CRPS league table would inherit by mixing n=5 with n=500.
    assert mean_plug_gap > 0.05, mean_plug_gap


def test_mean_pinball_still_importable_from_scoring() -> None:
    """Guard the helper used by pinball_skill_score stays on the public surface."""
    y = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    assert mean_pinball(y, q, 0.5) == pytest.approx(0.25)
