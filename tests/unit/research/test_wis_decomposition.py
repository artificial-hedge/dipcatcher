"""Weighted Interval Score + its exact 3-way decomposition (SOTA-05 gap G-2).

Bracher, Ray, Gneiting & Reich (2021), "Evaluating epidemic forecasts in an
interval format", *PLOS Comput. Biol.* 17(6):e1008618.

SYNTHETIC, seeded, deterministic — correctness evidence about estimator
behaviour, never market evidence. Nothing here is a live-trading or P&L claim.

The headline property under test is the *exact* additive identity
``dispersion + underprediction + overprediction == wis``, which turns "the score
got worse" into a diagnosable cause, plus ``WIS -> CRPS`` as the level grid
refines (the result that makes WIS a legitimate CRPS proxy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.calibration2 import (
    mean_wis,
    winkler_interval_score,
    wis_decomposition,
    wis_skill_score,
)
from quant_fund.metrics.scoring import crps_gaussian

SEED = 2026


# --- hand-computable K=1 cells ----------------------------------------------


def test_hand_computable_k1_observation_inside_the_interval() -> None:
    """y=0.5, interval [-1, 1] at alpha=0.1, median 0.

    w_1 = alpha/2 = 0.05, w_0 = 1/2, normaliser K + 1/2 = 1.5.
    ``IS_0.1`` = (u-l) = 2 (no penalty, y is inside), so the interval term is
    ``0.05 * 2 = 0.1``; the median term is ``0.5 * |0.5 - 0| = 0.25``.
    WIS = (0.1 + 0.25)/1.5 = 0.35/1.5 = 7/30.
    All of it is dispersion + overprediction; underprediction is exactly 0.
    """
    d = wis_decomposition(
        np.array([0.5]), np.array([[-1.0]]), np.array([[1.0]]), np.array([0.1]), np.array([0.0])
    )
    assert float(d["wis"][0]) == pytest.approx(0.35 / 1.5, abs=1e-12)
    assert float(d["dispersion"][0]) == pytest.approx(0.1 / 1.5, abs=1e-12)
    assert float(d["underprediction"][0]) == pytest.approx(0.0, abs=1e-15)
    assert float(d["overprediction"][0]) == pytest.approx(0.25 / 1.5, abs=1e-12)
    assert float(d["median_term"][0]) == pytest.approx(0.25 / 1.5, abs=1e-12)


def test_hand_computable_k1_observation_above_the_interval() -> None:
    """y=3, interval [-1, 1] at alpha=0.1, median 0 -> underprediction fires.

    ``IS`` = 2 + (2/0.1)(3-1) = 2 + 40 = 42; interval term ``0.05 * 42 = 2.1``.
    Median term ``0.5 * 3 = 1.5``. WIS = (2.1 + 1.5)/1.5 = 3.6/1.5 = 2.4.
    Underprediction = ``0.05 * 40 / 1.5`` = 4/3; overprediction = the median
    miss ``1.5/1.5`` = 1.0; dispersion = ``0.1/1.5``.
    """
    d = wis_decomposition(
        np.array([3.0]), np.array([[-1.0]]), np.array([[1.0]]), np.array([0.1]), np.array([0.0])
    )
    assert float(d["wis"][0]) == pytest.approx(3.6 / 1.5, abs=1e-12)
    assert float(d["underprediction"][0]) == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert float(d["overprediction"][0]) == pytest.approx(1.0, abs=1e-12)
    assert float(d["dispersion"][0]) == pytest.approx(0.1 / 1.5, abs=1e-12)


def test_hand_computable_k1_observation_below_the_interval() -> None:
    """y=-3 mirrors the case above into the overprediction component."""
    d = wis_decomposition(
        np.array([-3.0]), np.array([[-1.0]]), np.array([[1.0]]), np.array([0.1]), np.array([0.0])
    )
    assert float(d["wis"][0]) == pytest.approx(3.6 / 1.5, abs=1e-12)
    assert float(d["underprediction"][0]) == pytest.approx(0.0, abs=1e-15)
    assert float(d["overprediction"][0]) == pytest.approx(4.0 / 3.0 + 1.0, abs=1e-12)


def test_components_sum_to_the_total_exactly() -> None:
    """The work item's headline: ``dispersion + under + over == wis``.

    n=500 random observations, K=5 nested central intervals. The residual is
    pinned at exactly 0.0 (not merely small) because the decomposition is an
    algebraic regrouping of the same terms, and ``decomp_error`` reports it.
    """
    rng = np.random.default_rng(SEED)
    n = 500
    alphas = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
    mu = rng.normal(0.0, 1.0, size=n)
    sd = np.abs(rng.normal(1.0, 0.3, size=n)) + 0.2
    obs = rng.normal(mu, sd)
    lower = np.stack([mu + sd * norm.ppf(a / 2.0) for a in alphas], axis=1)
    upper = np.stack([mu + sd * norm.ppf(1.0 - a / 2.0) for a in alphas], axis=1)

    d = wis_decomposition(obs, lower, upper, alphas, mu)
    total = d["dispersion"] + d["underprediction"] + d["overprediction"]
    np.testing.assert_array_equal(total, d["wis"])
    np.testing.assert_array_equal(np.asarray(d["decomp_error"]), np.zeros(n))
    # Every component and the total are non-negative (they are sums of |.| terms).
    assert np.all(d["wis"] >= 0.0)
    assert np.all(d["dispersion"] >= 0.0)
    assert np.all(d["underprediction"] >= 0.0)
    assert np.all(d["overprediction"] >= 0.0)


def test_components_sum_to_the_total_without_a_median() -> None:
    """Median-free WIS still decomposes exactly, with normaliser K."""
    rng = np.random.default_rng(SEED + 1)
    n = 200
    alphas = np.array([0.1, 0.5])
    mu = rng.normal(size=n)
    sd = np.abs(rng.normal(1.0, 0.2, size=n)) + 0.3
    obs = rng.normal(mu, sd)
    lower = np.stack([mu + sd * norm.ppf(a / 2.0) for a in alphas], axis=1)
    upper = np.stack([mu + sd * norm.ppf(1.0 - a / 2.0) for a in alphas], axis=1)

    d = wis_decomposition(obs, lower, upper, alphas)
    np.testing.assert_array_equal(
        d["dispersion"] + d["underprediction"] + d["overprediction"], d["wis"]
    )
    np.testing.assert_array_equal(d["median_term"], np.zeros(n))
    # Hand-check the normaliser: dispersion = sum_k (alpha_k/2)(u_k-l_k)/K.
    k = float(alphas.size)
    expected_disp = np.sum((alphas / 2.0)[None, :] * (upper - lower), axis=1) / k
    np.testing.assert_allclose(d["dispersion"], expected_disp, atol=1e-12)


def test_wis_reduces_to_the_winkler_interval_score_at_k1() -> None:
    """With one level and no median, ``WIS * K == (alpha/2) * IS_alpha``.

    Cross-checks the new aggregation against the pre-existing
    :func:`winkler_interval_score` so the two interval scores cannot drift apart.
    """
    rng = np.random.default_rng(SEED + 2)
    n = 300
    alpha = 0.1
    lo = rng.normal(-1.0, 0.3, size=n)
    hi = lo + np.abs(rng.normal(2.0, 0.5, size=n)) + 0.1
    obs = rng.normal(0.0, 1.5, size=n)

    wink = winkler_interval_score(lo, hi, obs, alpha=alpha)
    d = wis_decomposition(obs, lo.reshape(-1, 1), hi.reshape(-1, 1), np.array([alpha]))
    np.testing.assert_allclose(
        np.asarray(d["wis"]), (alpha / 2.0) * np.asarray(wink["score"]), atol=1e-12
    )


def test_wis_converges_to_the_crps_as_the_level_grid_refines() -> None:
    r"""``WIS -> CRPS`` as ``K -> inf`` (BRGR 2021) — the reason WIS is usable.

    Gaussian predictive ``N(0,1)``, so the target is the closed-form mean CRPS.
    Measured ratio to the closed form across the level grid, monotone toward 1:

    K=5 -> 1.1159, K=11 -> 1.0583, K=23 -> 1.0293, K=47 -> 1.0147,
    K=95 -> 1.0073, K=191 -> 1.0037.
    """
    rng = np.random.default_rng(99)
    n = 20000
    mu = np.zeros(n)
    sd = np.ones(n)
    obs = rng.normal(0.0, 1.0, size=n)
    target = float(np.mean(crps_gaussian(obs, mu, sd)))

    ratios = []
    for k in (5, 11, 23, 47, 95, 191):
        taus = np.linspace(0.0, 1.0, 2 * k + 2)[1:-1]
        alphas = np.sort(2.0 * taus[taus < 0.5])[::-1]
        lower = np.stack([mu + sd * norm.ppf(a / 2.0) for a in alphas], axis=1)
        upper = np.stack([mu + sd * norm.ppf(1.0 - a / 2.0) for a in alphas], axis=1)
        mw = mean_wis(obs, lower, upper, alphas, mu)
        ratios.append(mw["mean_wis"] / target)

    # Monotonically improving and converging on the CRPS.
    assert all(abs(r - 1.0) > abs(rn - 1.0) for r, rn in zip(ratios, ratios[1:], strict=False))
    assert ratios[0] == pytest.approx(1.1159, abs=0.005)
    assert ratios[-1] == pytest.approx(1.0, abs=0.01)


def test_decomposition_attributes_the_right_component() -> None:
    """A forecast that is far too low shows up as underprediction, not dispersion.

    This is the diagnostic value of the decomposition: two forecasts with the
    same total can be told apart by *why* they are bad.
    """
    n = 200
    alphas = np.array([0.1, 0.5])
    obs = np.full(n, 5.0)
    # Forecast A: correct width, badly shifted low -> location failure.
    lo_a = np.full((n, 2), -1.0)
    hi_a = np.full((n, 2), 1.0)
    # Forecast B: correctly centred on the observation but absurdly wide ->
    # pure dispersion failure.
    lo_b = np.full((n, 2), -50.0)
    hi_b = np.full((n, 2), 50.0)
    med = np.zeros(n)

    da = wis_decomposition(obs, lo_a, hi_a, alphas, med)
    db = wis_decomposition(obs, lo_b, hi_b, alphas, med)
    assert float(np.mean(da["underprediction"])) > float(np.mean(da["dispersion"]))
    assert float(np.mean(db["underprediction"])) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(db["overprediction"])) == pytest.approx(
        float(np.mean(db["median_term"])), abs=1e-12
    )
    assert float(np.mean(db["dispersion"])) > float(np.mean(da["dispersion"]))


# --- fail-closed contracts --------------------------------------------------


def test_wis_fail_closed_on_malformed_inputs() -> None:
    obs = np.array([0.5])
    lo = np.array([[-1.0]])
    hi = np.array([[1.0]])
    alphas = np.array([0.1])
    with pytest.raises(ValueError, match="non-empty"):
        wis_decomposition(np.array([]), np.zeros((0, 1)), np.zeros((0, 1)), alphas)
    with pytest.raises(ValueError, match="2-D"):
        wis_decomposition(obs, [-1.0], [[1.0]], alphas)
    with pytest.raises(ValueError, match="shape mismatch"):
        wis_decomposition(obs, lo, np.array([[1.0], [2.0]]), alphas)
    with pytest.raises(ValueError, match="at least one interval"):
        wis_decomposition(obs, np.zeros((1, 0)), np.zeros((1, 0)), np.array([]))
    with pytest.raises(ValueError, match="must match K"):
        wis_decomposition(obs, lo, hi, np.array([0.1, 0.5]))
    with pytest.raises(ValueError, match="strictly inside"):
        wis_decomposition(obs, lo, hi, np.array([1.5]))
    with pytest.raises(ValueError, match="strictly inside"):
        wis_decomposition(obs, lo, hi, np.array([0.0]))
    with pytest.raises(ValueError, match="crossed interval"):
        wis_decomposition(obs, np.array([[1.0]]), np.array([[-1.0]]), alphas)
    with pytest.raises(ValueError, match="median length"):
        wis_decomposition(obs, lo, hi, alphas, np.array([0.0, 1.0]))


def test_wis_propagates_non_finite_entries_per_observation() -> None:
    """One bad observation must not poison the whole panel."""
    obs = np.array([0.5, np.nan])
    lo = np.array([[-1.0], [-1.0]])
    hi = np.array([[1.0], [1.0]])
    alphas = np.array([0.1])
    d = wis_decomposition(obs, lo, hi, alphas, np.array([0.0, 0.0]))
    assert float(d["wis"][0]) == pytest.approx(0.35 / 1.5, abs=1e-12)
    assert math.isnan(float(d["wis"][1]))
    # mean_wis drops the non-finite observation consistently across components.
    mw = mean_wis(obs, lo, hi, alphas, np.array([0.0, 0.0]))
    assert mw["n_valid"] == 1.0
    assert mw["mean_wis"] == pytest.approx(0.35 / 1.5, abs=1e-12)
    assert mw["mean_wis"] == pytest.approx(
        mw["mean_dispersion"] + mw["mean_underprediction"] + mw["mean_overprediction"], abs=1e-12
    )


def test_mean_wis_all_nan_is_fail_closed() -> None:
    obs = np.array([np.nan, np.nan])
    lo = np.array([[-1.0], [-1.0]])
    hi = np.array([[1.0], [1.0]])
    mw = mean_wis(obs, lo, hi, np.array([0.1]))
    assert math.isnan(mw["mean_wis"])
    assert mw["n_valid"] == 0.0


# --- WIS skill score (work item 5, interval-format version) -----------------


def test_wis_skill_score_is_zero_for_an_identical_forecast() -> None:
    rng = np.random.default_rng(SEED + 3)
    n = 200
    alphas = np.array([0.1, 0.5])
    mu = rng.normal(size=n)
    sd = np.abs(rng.normal(1.0, 0.2, size=n)) + 0.2
    obs = rng.normal(mu, sd)
    lo = np.stack([mu + sd * norm.ppf(a / 2.0) for a in alphas], axis=1)
    hi = np.stack([mu + sd * norm.ppf(1.0 - a / 2.0) for a in alphas], axis=1)
    assert wis_skill_score(obs, lo, hi, alphas, lo, hi, mu, mu) == pytest.approx(0.0, abs=1e-12)


def test_wis_skill_score_is_negative_when_worse_and_positive_when_better() -> None:
    rng = np.random.default_rng(SEED + 4)
    n = 300
    alphas = np.array([0.1, 0.5])
    obs = rng.normal(0.0, 1.0, size=n)
    good_mu = np.zeros(n)
    good_sd = np.ones(n)
    lo_g = np.stack([good_mu + good_sd * norm.ppf(a / 2.0) for a in alphas], axis=1)
    hi_g = np.stack([good_mu + good_sd * norm.ppf(1.0 - a / 2.0) for a in alphas], axis=1)
    # A uselessly wide forecast: strictly worse than the calibrated one.
    lo_w = lo_g - 5.0
    hi_w = hi_g + 5.0

    assert wis_skill_score(obs, lo_w, hi_w, alphas, lo_g, hi_g, good_mu, good_mu) < 0.0
    assert wis_skill_score(obs, lo_g, hi_g, alphas, lo_w, hi_w, good_mu, good_mu) > 0.0
