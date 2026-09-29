"""Propriety + boundary tests for ``threshold_energy_score`` (SOTA-05 F-01).

SYNTHETIC, seeded, deterministic — correctness evidence about estimator
behaviour, never market evidence. All figures here are research diagnostics;
nothing is a live-trading or P&L claim.

The function's *arithmetic* is deliberately NOT changed by this lane: existing
tests pin it and sealed receipts may reference it, so altering the numbers
would invalidate immutable evidence (AGENTS.md honesty contract rule #4). These
tests pin the measured improperness at and above ``weight = sqrt(2)``, pin the
safe behaviour below it, and pin the new ``RuntimeWarning`` + docstring contract.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from quant_fund.metrics.energy_score import (
    THRESHOLD_WEIGHT_PROPRIETY_BOUND,
    energy_score,
    threshold_energy_score,
)

SEED = 20260928
_MEMBERS = 60
_REPS = 400


@contextmanager
def _no_runtime_warning() -> Iterator[list[warnings.WarningMessage]]:
    """Collect warnings without failing; ``pytest.warns(None)`` was removed in pytest 9."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _expected_es_sigma(sigma: float, weight: float, threshold: float = 1.0) -> float:
    """Monte-Carlo ``E[ES_w]`` for obs ~ N(0, I_d), forecast ~ N(0, sigma^2 I_d).

    d = 2, matching the measured table in docs/SOTA/05-scoring-rules.md §3 F-01.
    """
    rng = np.random.default_rng(SEED)
    vals: list[float] = []
    for _ in range(_REPS):
        ens = rng.normal(0.0, sigma, size=(_MEMBERS, 2))
        obs = rng.normal(0.0, 1.0, size=2)
        # The RuntimeWarning is diagnostic-only (arithmetic is unchanged); it is
        # asserted separately so the Monte-Carlo loop stays fast and quiet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals.append(threshold_energy_score(ens, obs, threshold=threshold, weight=weight))
    return float(np.mean(vals))


def test_propriety_bound_is_exactly_sqrt_two() -> None:
    """The exported constant must be sqrt(2), the analytic breakdown weight."""
    assert math.sqrt(2.0) == THRESHOLD_WEIGHT_PROPRIETY_BOUND


def test_below_threshold_weight_is_proper_interior_minimum() -> None:
    """``weight < sqrt(2)``: E[ES_w] has an interior minimum, not monotone decay.

    Pinning weight = 1.0 (the shipped default, safe) and weight = 1.2: the
    expectation must dip and then rise as dispersion grows, i.e. inflating
    variance is *penalised*, which is the whole point of a sharpness-aware score.
    """
    for weight in (1.0, 1.2):
        sigmas = (0.5, 1.0, 2.0, 4.0)
        means = [_expected_es_sigma(s, weight) for s in sigmas]
        interior = min(means)
        # The minimum sits strictly inside the grid (not at the widest sigma)
        # and the widest forecast is clearly worse than the best one.
        assert means.index(interior) < len(means) - 1, f"weight={weight}: {means}"
        assert means[-1] > interior + 1e-6, f"weight={weight}: {means}"


def test_at_sqrt_two_score_decreases_monotonically_with_dispersion() -> None:
    """``weight = sqrt(2)`` is the *boundary*: already monotone decreasing.

    The leading coefficient w - w^2/sqrt(2) vanishes exactly here, so the
    residual finite-threshold correction dominates and the score drifts down as
    sigma grows — sqrt(2) is NOT a safe value, it is the edge of the domain.
    """
    w = THRESHOLD_WEIGHT_PROPRIETY_BOUND
    coarse = _expected_es_sigma(0.5, w)
    fine = _expected_es_sigma(4.0, w)
    assert fine < coarse, f"weight=sqrt(2): sigma=0.5 -> {coarse}, sigma=4 -> {fine}"


@pytest.mark.parametrize("weight", [1.5, 2.0, 3.0])
def test_above_threshold_weight_is_improper_rewards_variance_inflation(
    weight: float,
) -> None:
    """``weight > sqrt(2)``: E[ES_w] decreases as dispersion grows, crossing zero.

    This is the CRITICAL defect from the brief, pinned empirically so it can
    never be silently "fixed" into a different improper function, and so the
    escalation in docs/SOTA/24-metrics-findings.md is reproducible.
    """
    low = _expected_es_sigma(0.5, weight)
    high = _expected_es_sigma(8.0, weight)
    assert high < low, f"weight={weight}: sigma=0.5 -> {low}, sigma=8 -> {high}"
    # A proper energy score is bounded below by 0 in the population; this one
    # goes negative, which is impossible for a proper rule.
    assert high < 0.0, f"weight={weight}: E[ES_w] at sigma=8 = {high}"


def test_measured_values_match_the_brief_within_monte_carlo_error() -> None:
    """Pin the headline cell: weight=2.0, d=2, 60 members, 400 reps, seed 20260928.

    The brief measured 1.386 -> -8.007 as sigma goes 0.5 -> 8. Our Monte-Carlo
    lands within 0.1 of both ends (independent draw order, same seed/size),
    which pins the magnitude without over-fitting to exact RNG stream ordering.
    """
    low = _expected_es_sigma(0.5, 2.0)
    high = _expected_es_sigma(8.0, 2.0)
    assert low == pytest.approx(1.386, abs=0.15)
    assert high == pytest.approx(-8.007, abs=0.15)


def test_breakdown_weight_is_dimension_free() -> None:
    """2a/b == sqrt(2) for d = 1..16, where a = E||Z||, b = E||Z - Z'||.

    The threshold does not depend on dimension because ``a`` cancels in the
    ratio; verified numerically as the brief claims.
    """
    for d in (1, 2, 3, 4, 8, 16):
        rng = np.random.default_rng(SEED + d)
        z = rng.normal(0.0, 1.0, size=(20000, d))
        zp = rng.normal(0.0, 1.0, size=(20000, d))
        a = float(np.mean(np.linalg.norm(z, axis=1)))
        b = float(np.mean(np.linalg.norm(z - zp, axis=1)))
        assert 2.0 * a / b == pytest.approx(math.sqrt(2.0), abs=0.01), f"d={d}"


def test_control_plain_energy_score_is_proper() -> None:
    """Positive control: the unweighted ``energy_score`` has an interior minimum.

    Same Monte-Carlo cell, so the improperness is specific to the threshold
    weighting and not an artefact of the harness.
    """
    rng = np.random.default_rng(SEED)
    means = []
    for sigma in (0.5, 1.0, 1.5, 2.0):
        vals = []
        for _ in range(_REPS):
            ens = rng.normal(0.0, sigma, size=(_MEMBERS, 2))
            obs = rng.normal(0.0, 1.0, size=2)
            vals.append(energy_score(ens, obs))
        means.append(float(np.mean(vals)))
    assert means.index(min(means)) < len(means) - 1, means
    assert means[-1] > min(means)


def test_runtime_warning_emitted_at_or_above_sqrt_two() -> None:
    """``weight >= sqrt(2)`` must emit a RuntimeWarning naming the breakdown."""
    rng = np.random.default_rng(0)
    ens = rng.normal(0.0, 1.0, size=(10, 2))
    obs = np.zeros(2)
    with pytest.warns(RuntimeWarning, match="IMPROPER"):
        threshold_energy_score(ens, obs, threshold=1.0, weight=2.0)
    with pytest.warns(RuntimeWarning, match="sqrt\\(2\\)"):
        threshold_energy_score(ens, obs, threshold=1.0, weight=THRESHOLD_WEIGHT_PROPRIETY_BOUND)


def test_no_warning_below_sqrt_two() -> None:
    """Sub-threshold weights are the documented proper domain: no warning."""
    rng = np.random.default_rng(0)
    ens = rng.normal(0.0, 1.0, size=(10, 2))
    obs = np.zeros(2)
    with _no_runtime_warning() as caught:
        threshold_energy_score(ens, obs, threshold=1.0, weight=1.3)
        threshold_energy_score(ens, obs, threshold=1.0, weight=1.0)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == []


def test_arithmetic_unchanged_by_the_warning() -> None:
    """The warning must be diagnostic-only: values identical to the pinned formula.

    Guards against anyone "fixing" F-01 by silently changing the numbers, which
    would invalidate sealed receipts. Same hand-computable case as
    tests/unit/core/test_energy_score.py::test_hand_computable_threshold_weighted_3x2.
    """
    ensemble = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    obs = np.zeros(2)
    expected = (2.0 - 2.0 * np.sqrt(2.0)) / 3.0
    with pytest.warns(RuntimeWarning):
        got = threshold_energy_score(ensemble, obs, threshold=0.5, weight=2.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_docstring_no_longer_claims_unconditional_propriety() -> None:
    """The false claim ("remains a proper scoring rule") must be gone.

    The brief flags the docstring as the *more dangerous* half of F-01: it would
    let a reviewer wave an improper score through. Pin the corrected contract.
    """
    doc = threshold_energy_score.__doc__
    assert doc is not None
    lowered = doc.lower()
    assert "remains a proper scoring rule" not in lowered
    assert "sqrt(2)" in lowered or "√2" in lowered
    assert "improper" in lowered
