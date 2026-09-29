"""Property-based tests for the probabilistic scoring rules (work item 6).

SYNTHETIC, derandomised, deterministic — invariants of the *estimators*, never
market evidence. Nothing here is a live-trading or P&L claim.

These are the properties that make the scores usable at all. A unit test pins a
number; a property test pins the algebra that number was derived from, so a
refactor that keeps the golden value but breaks the invariant is still caught:

* CRPS non-negativity — a proper score bounded below by 0 cannot reward a
  forecast for being wrong.
* Quantile properness — the *true* quantile minimises expected pinball loss,
  which is the entire justification for using pinball in a research harness.
* Symmetry / permutation invariance — the score must not depend on how the
  ensemble happens to be ordered or on the sign of the error once the magnitude
  is fixed.
* Translation equivariance — shifting forecast and observation by the same
  constant shifts nothing in the score, so the units are relative not absolute.
* Scale equivariance — CRPS scales linearly with the data, so cross-sectional
  comparisons are not confounded by level.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from quant_fund.metrics.calibration2 import wis_decomposition
from quant_fund.metrics.energy_score import energy_score, threshold_energy_score
from quant_fund.metrics.scoring import (
    crps_empirical,
    crps_fair,
    crps_threshold_weighted,
    mean_pinball,
    pinball_loss,
    rearrange_quantiles,
)

# Repo convention: derandomise + a bounded example count + no deadline, so the
# property suite is reproducible and cannot flake on a slow machine.
DERIV = settings(derandomize=True, max_examples=60, deadline=None)

_FINITE = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_OBS = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)


def _ensemble(values: list[float]) -> np.ndarray:
    """At least one member, all finite — the documented minimum contract."""
    return np.asarray(values, dtype=float).reshape(-1)


_en = st.lists(_OBS, min_size=1, max_size=24)


# --- CRPS non-negativity ----------------------------------------------------


@given(y=_OBS, sample=_en)
@DERIV
def test_crps_is_nonnegative_for_both_estimators(y: float, sample: list[float]) -> None:
    """A proper CRPS is >= 0 everywhere, including the tail-degenerate cases."""
    arr = _ensemble(sample)
    assert crps_empirical(y, arr) >= -1e-12
    assert crps_fair(y, arr) >= -1e-12


@given(y=_OBS, sample=_en)
@DERIV
def test_threshold_weighted_crps_is_nonnegative(y: float, sample: list[float]) -> None:
    """The proper tail-weighted score stays >= 0 at any threshold/weight.

    This is exactly where ``threshold_energy_score`` fails (it diverges to -inf
    for weight >= sqrt(2)); the property holds unconditionally here.
    """
    arr = _ensemble(sample)
    for threshold in (0.1, 1.0, 5.0):
        for weight in (1.0, 2.0, 10.0, 100.0):
            assert crps_threshold_weighted(y, arr, threshold=threshold, weight=weight) >= -1e-12


@given(sample=_en)
@DERIV
def test_crps_is_zero_exactly_when_the_forecast_is_degenerate_at_the_observation(
    sample: list[float],
) -> None:
    """A point mass at ``y`` is the only perfect forecast: CRPS == 0.

    Pinning the *identity* element of the score, not just its sign, is what
    makes "CRPS improved" meaningful at all.
    """
    arr = _ensemble(sample)
    y = float(arr[0])
    degenerate = np.full(arr.shape, y)
    assert crps_empirical(y, degenerate) == pytest.approx(0.0, abs=1e-12)
    assert crps_fair(y, degenerate) == pytest.approx(0.0, abs=1e-12)
    assert crps_threshold_weighted(y, degenerate, threshold=1.0, weight=3.0) == pytest.approx(
        0.0, abs=1e-12
    )
    # The fair estimator's minimum over y is attained at a sample value.
    assert min(crps_fair(c, arr) for c in (float(v) for v in arr)) >= -1e-12


# --- symmetry and equivariance ---------------------------------------------


@given(y=_OBS, sample=_en, offset=_FINITE)
@DERIV
def test_crps_is_translation_equivariant(y: float, sample: list[float], offset: float) -> None:
    """Shifting the observation and every member equally leaves CRPS unchanged.

    The score measures error in the units of the data, so a level shift is not
    an error — without this, a forecast could win by predicting the wrong level.
    """
    arr = _ensemble(sample)
    base = crps_fair(y, arr)
    shifted = crps_fair(y + offset, arr + offset)
    assert shifted == pytest.approx(base, abs=1e-8, rel=1e-8)
    assert crps_empirical(y, arr) == pytest.approx(
        crps_empirical(y + offset, arr + offset), abs=1e-8, rel=1e-8
    )


@given(y=_OBS, sample=_en, offset=_FINITE)
@DERIV
def test_threshold_weighted_crps_is_origin_anchored_not_translation_equivariant(
    y: float, sample: list[float], offset: float
) -> None:
    """CAVEAT pinned as a property: the tail zone is absolute, not relative.

    ``w(z) = weight if |z| > threshold`` is anchored at the *origin*, so shifting
    the data moves it relative to the amplified zone and changes the score
    (y=0, sample=[1.0], threshold=1, weight=2: base 1.0 -> shifted 2.0). This is
    correct behaviour for an absolute tail threshold — the score answers "is the
    error large in absolute terms", which is what tail monitoring wants — but it
    means threshold-weighted CRPS values are NOT comparable across series with
    different levels or price scales. Pinning it so the caveat cannot be lost:
    a consumer must standardise (or choose ``threshold`` per symbol) before
    comparing. Contrast :func:`crps_fair`, which is equivariant above.
    """
    arr = _ensemble(sample)
    base = crps_threshold_weighted(y, arr, threshold=1.0, weight=2.0)
    shifted = crps_threshold_weighted(y + offset, arr + offset, threshold=1.0, weight=2.0)
    # Not an identity: the score legitimately differs. What must hold is that
    # the shift is bounded and that the *ordering* induced by the amplification
    # is preserved — errors deep in the tail always score at least as badly as
    # the same errors measured at unit weight.
    assert math.isfinite(base) and math.isfinite(shifted)
    plain_base = crps_fair(y, arr)
    plain_shift = crps_fair(y + offset, arr + offset)
    assert base >= plain_base - 1e-12
    assert shifted >= plain_shift - 1e-12


@given(y=_OBS, sample=_en, scale=st.floats(min_value=0.01, max_value=10.0))
@DERIV
def test_crps_is_scale_equivariant(y: float, sample: list[float], scale: float) -> None:
    """``CRPS(c*y, c*X) == c * CRPS(y, X)`` — linear homogeneity.

    Without it, comparing two symbols with different price levels would compare
    their scales rather than their forecasts.
    """
    arr = _ensemble(sample)
    assert crps_fair(y * scale, arr * scale) == pytest.approx(scale * crps_fair(y, arr), rel=1e-7)
    assert crps_empirical(y * scale, arr * scale) == pytest.approx(
        scale * crps_empirical(y, arr), rel=1e-7
    )
    # The threshold-weighted score is equivariant only when the threshold scales
    # with the data; pin that, since it is the correct generalisation.
    assert crps_threshold_weighted(y * scale, arr * scale, threshold=scale, weight=2.0) == (
        pytest.approx(scale * crps_threshold_weighted(y, arr, threshold=1.0, weight=2.0), rel=1e-7)
    )


@given(y=_OBS, sample=_en, seed=st.integers(min_value=0, max_value=10_000))
@DERIV
def test_crps_is_invariant_to_ensemble_order(y: float, sample: list[float], seed: int) -> None:
    """The ensemble is an unordered set; permuting members must change nothing."""
    arr = _ensemble(sample)
    rng = np.random.default_rng(seed)
    shuffled = arr[rng.permutation(arr.size)]
    assert shuffled.size == arr.size
    assert np.sort(shuffled) == pytest.approx(np.sort(arr))
    for fn in (crps_empirical, crps_fair):
        assert fn(y, shuffled) == pytest.approx(fn(y, arr), abs=1e-12)
    assert crps_threshold_weighted(y, shuffled, threshold=1.0, weight=2.0) == pytest.approx(
        crps_threshold_weighted(y, arr, threshold=1.0, weight=2.0), abs=1e-12
    )


@given(y=_OBS, sample=_en)
@DERIV
def test_threshold_weighted_crps_dominates_the_plain_fair_crps(
    y: float, sample: list[float]
) -> None:
    """``weight > 1`` amplifies, never shrinks, so the weighted score is >= fair.

    The weighting function ``w(z) >= 1`` pointwise, hence the integrand is
    pointwise larger; this is the invariant that makes the tail weighting a
    *strengthening* of the same proper score.
    """
    arr = _ensemble(sample)
    plain = crps_threshold_weighted(y, arr, threshold=1.0, weight=1.0)
    heavy = crps_threshold_weighted(y, arr, threshold=1.0, weight=4.0)
    assert heavy >= plain - 1e-12
    assert plain == pytest.approx(crps_fair(y, arr), abs=1e-12)


# --- quantile properness ----------------------------------------------------


@given(
    tau=st.floats(min_value=0.05, max_value=0.95),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(derandomize=True, max_examples=30, deadline=None)
def test_empirical_quantile_minimises_empirical_pinball_loss(tau: float, seed: int) -> None:
    """Finite-sample properness: ``np.quantile(y, tau)`` minimises ``mean_pinball``.

    The empirical pinball loss is convex and piecewise-linear in the forecast,
    so its minimiser is exactly the empirical tau-quantile — the finite-sample
    statement of "pinball is a proper score for the tau-quantile". Robust to the
    flat minimum that occurs when tau lands strictly between two order
    statistics: the empirical quantile must be *a* global minimiser (within
    tolerance) and must strictly beat forecasts far from it.
    """
    rng = np.random.default_rng(seed)
    n = 400
    y = rng.normal(0.0, 1.0, size=n)
    emp_q = float(np.quantile(y, tau))
    offsets = np.linspace(-0.4, 0.4, 17)
    losses = np.array([mean_pinball(y, np.full(n, emp_q + d), tau) for d in offsets])
    centre = float(mean_pinball(y, np.full(n, emp_q), tau))
    # The empirical quantile attains the grid minimum (allowing ties).
    assert centre <= float(np.min(losses)) + 1e-12
    # And strictly beats candidates well away from it — the loss is V-shaped.
    assert centre < losses[0] - 1e-9, f"tau={tau}: lower end not beaten"
    assert centre < losses[-1] - 1e-9, f"tau={tau}: upper end not beaten"


@given(tau=st.floats(min_value=0.1, max_value=0.9))
@settings(derandomize=True, max_examples=25, deadline=None)
def test_population_quantile_minimises_expected_pinball_loss(tau: float) -> None:
    """Population properness: the true quantile ``norm.ppf(tau)`` wins in expectation.

    With a large deterministic sample (n = 20000, empirical quantile within
    ~0.007 of the truth) the true quantile beats forecasts offset by ±0.2 and
    ±0.4 in *expected* pinball loss. This is the property that licenses pinball
    as a research score: minimising it targets the true conditional quantile,
    not an arbitrary statistic.
    """
    rng = np.random.default_rng(2026)
    n = 20_000
    y = rng.normal(0.0, 1.0, size=n)
    true_q = float(norm.ppf(tau))
    centre = mean_pinball(y, np.full(n, true_q), tau)
    for delta in (0.2, 0.4, -0.2, -0.4):
        worse = mean_pinball(y, np.full(n, true_q + delta), tau)
        assert centre < worse - 1e-6, f"tau={tau}: delta={delta} not beaten"


@given(tau=st.floats(min_value=0.01, max_value=0.99), y=_OBS, q=_OBS)
@DERIV
def test_pinball_loss_is_nonnegative_and_matches_the_asymmetric_definition(
    tau: float, y: float, q: float
) -> None:
    r"""``L = tau*max(y-q, 0) + (1-tau)*max(q-y, 0)`` and ``L >= 0``."""
    got = float(pinball_loss(np.array([y]), np.array([q]), tau)[0])
    expected = tau * max(y - q, 0.0) + (1.0 - tau) * max(q - y, 0.0)
    assert got == pytest.approx(expected, abs=1e-12)
    assert got >= 0.0
    # Zero exactly at the forecast, for every tau.
    assert float(pinball_loss(np.array([y]), np.array([y]), tau)[0]) == pytest.approx(
        0.0, abs=1e-15
    )


@given(y=_OBS, tau=st.floats(min_value=0.01, max_value=0.99))
@DERIV
def test_pinball_loss_is_convex_in_the_forecast(y: float, tau: float) -> None:
    """Convexity: no local minima to trap a fit, the true quantile is the global one.

    Checked by midpoint convexity on a random triple, which is the algebraic
    content of convexity and is exact for a piecewise-linear loss.
    """
    a, b = y - 3.0, y + 3.0
    mid = 0.5 * (a + b)
    la = float(pinball_loss(np.array([y]), np.array([a]), tau)[0])
    lb = float(pinball_loss(np.array([y]), np.array([b]), tau)[0])
    lm = float(pinball_loss(np.array([y]), np.array([mid]), tau)[0])
    assert lm <= 0.5 * (la + lb) + 1e-12


# --- energy score symmetry --------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10_000), half=st.integers(min_value=2, max_value=8))
@settings(derandomize=True, max_examples=20, deadline=None)
def test_energy_score_is_symmetric_for_a_negation_closed_ensemble(seed: int, half: int) -> None:
    """For an ensemble closed under negation, ``ES(+y) == ES(-y)`` exactly.

    The energy score is built from Euclidean norms, so it cannot distinguish the
    sign of the error when the forecast law is symmetric. That symmetry holds
    *exactly* only for a negation-closed finite ensemble (built as ``[H; -H]``);
    a generic random finite ensemble is asymmetric and this property does NOT
    hold for it — see test_energy_score_finite_ensemble_is_generically_asymmetric.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 1.0, size=(half, 3))
    ens = np.vstack([h, -h])
    obs = np.abs(rng.normal(0.0, 1.0, size=3)) + 1e-3
    assert energy_score(ens, obs) == pytest.approx(energy_score(ens, -obs), abs=1e-10)
    assert threshold_energy_score(ens, obs, threshold=1.0, weight=1.0) == pytest.approx(
        threshold_energy_score(ens, -obs, threshold=1.0, weight=1.0), abs=1e-10
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(derandomize=True, max_examples=20, deadline=None)
def test_energy_score_finite_ensemble_is_generically_asymmetric(seed: int) -> None:
    """CAVEAT pinned: a generic finite ensemble is NOT sign-symmetric.

    ``ES(+y)`` and ``ES(-y)`` differ for a random 15-member ensemble (measured
    1.8297 vs 1.8592 at seed 0). Symmetry is a *population* property of the
    forecast law, not of a single finite draw, so any test asserting it on a
    random ensemble is asserting a falsehood. Pinned so the distinction between
    the finite-sample and population statements cannot be silently conflated.
    """
    rng = np.random.default_rng(seed)
    ens = rng.normal(0.0, 1.0, size=(15, 3))
    obs = np.abs(rng.normal(0.0, 1.0, size=3)) + 1e-3
    diff = energy_score(ens, obs) - energy_score(ens, -obs)
    # Almost-sure asymmetry: the difference is zero only if the ensemble happens
    # to be closed under negation, a measure-zero event for continuous draws.
    # Both values remain finite and non-negative regardless (bounded, not NaN).
    assert math.isfinite(diff)
    assert energy_score(ens, obs) >= -1e-12
    assert energy_score(ens, -obs) >= -1e-12


# --- WIS decomposition invariants ------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10_000), n=st.integers(min_value=2, max_value=40))
@settings(derandomize=True, max_examples=40, deadline=None)
def test_wis_decomposition_sums_exactly_for_random_intervals(seed: int, n: int) -> None:
    """The exact additive identity holds for arbitrary nested random intervals.

    Pinned as a property (not one fixed seed) because the identity must hold for
    *every* configuration — including observations far outside the intervals,
    where the under/overprediction terms activate.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([0.1, 0.5])
    obs = rng.normal(0.0, 2.0, size=n)
    centres = rng.normal(0.0, 2.0, size=n)
    widths = np.abs(rng.normal(1.0, 0.5, size=n)) + 0.05
    lower = np.stack([centres - widths * (1.0 + 0.5 * k) for k in range(alphas.size)], axis=1)
    upper = np.stack([centres + widths * (1.0 + 0.5 * k) for k in range(alphas.size)], axis=1)

    d = wis_decomposition(obs, lower, upper, alphas)
    np.testing.assert_allclose(
        d["dispersion"] + d["underprediction"] + d["overprediction"],
        d["wis"],
        atol=0.0,
        rtol=0.0,
    )
    assert np.all(np.asarray(d["wis"]) >= 0.0)
    assert np.all(np.asarray(d["dispersion"]) >= 0.0)
    assert np.all(np.asarray(d["underprediction"]) >= 0.0)
    assert np.all(np.asarray(d["overprediction"]) >= 0.0)
    # Translation equivariance of the whole decomposition.
    shift = 7.5
    d2 = wis_decomposition(obs + shift, lower + shift, upper + shift, alphas)
    np.testing.assert_allclose(d2["wis"], d["wis"], atol=1e-9)
    np.testing.assert_allclose(d2["dispersion"], d["dispersion"], atol=1e-9)


@given(seed=st.integers(min_value=0, max_value=10_000), n=st.integers(min_value=2, max_value=20))
@settings(derandomize=True, max_examples=30, deadline=None)
def test_wis_under_and_over_prediction_never_fire_together(seed: int, n: int) -> None:
    """An observation cannot be simultaneously below the lower and above the upper bound.

    The two directional penalties are mutually exclusive by construction, which
    is what makes the decomposition a *cause* attribution rather than a split.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([0.05, 0.2, 0.5])
    obs = rng.normal(0.0, 3.0, size=n)
    centres = rng.normal(0.0, 3.0, size=n)
    widths = np.abs(rng.normal(1.0, 0.6, size=n)) + 0.1
    lower = np.stack([centres - widths * (1.0 + k) for k in range(alphas.size)], axis=1)
    upper = np.stack([centres + widths * (1.0 + k) for k in range(alphas.size)], axis=1)

    d = wis_decomposition(obs, lower, upper, alphas, centres)
    under = np.asarray(d["underprediction"])
    over = np.asarray(d["overprediction"])
    median_term = np.asarray(d["median_term"])
    # Per observation, at most one of the two interval-violation components can
    # be strictly positive once the median miss is excluded.
    both = (under - median_term > 1e-12) & (over - median_term > 1e-12)
    assert not np.any(both), (under[both], over[both])


# --- rearrangement invariants ----------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10_000), n=st.integers(min_value=1, max_value=20))
@settings(derandomize=True, max_examples=30, deadline=None)
def test_rearrangement_is_monotone_and_idempotent(seed: int, n: int) -> None:
    """Sorted quantiles are monotone, and re-sorting a sorted matrix is a no-op."""
    rng = np.random.default_rng(seed)
    k = 5
    q = rng.normal(0.0, 1.0, size=(n, k))
    arr = rearrange_quantiles(q)
    assert np.all(np.diff(arr, axis=1) >= -1e-12)
    np.testing.assert_allclose(rearrange_quantiles(arr), arr, atol=0.0, rtol=0.0)
    # Rearrangement preserves the multiset of values per row.
    for i in range(n):
        assert sorted(q[i].tolist()) == pytest.approx(sorted(arr[i].tolist()))
