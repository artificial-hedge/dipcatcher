"""Proper-score decompositions + discrete/count proper-score canon.

SYNTHETIC correctness tests only (AGENTS.md honesty contract #2): seeded
propriety (minimization at the truth), exact decomposition identities
(Bröcker 2012 ensemble CRPS; Kolassa 2016 discrete CRPS; Murphy/Bröcker
Brier REL-RES+UNC), determinism pinned, and fail-closed edges.
Research-diagnostic only — live_pnl_claim=false; never market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from quant_fund.metrics.calibration2 import murphy_decomposition
from quant_fund.metrics.probability import brier_score, log_loss
from quant_fund.metrics.score_decomposition import (
    brier_decomposition,
    brier_decomposition_binary,
    brier_multiclass,
    broecker_ensemble_crps_decomposition,
    fair_ensemble_crps,
    kolassa_crps_decomposition,
    log_score_discrete,
    mean_brier_multiclass,
    mean_log_score_discrete,
    mean_ranked_probability_score,
    mean_spherical_score,
    ranked_probability_score,
    sharpness_decomposition,
    spherical_score,
)
from quant_fund.metrics.scoring import crps_empirical, crps_gaussian, mean_crps_gaussian
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

# Shared SYNTHETIC truth on K=5 ordered categories (all entries > 0).
TRUTH_K5 = np.array([0.10, 0.20, 0.35, 0.25, 0.10])


def _expected_score_under_truth(score_fn, pmf_row: np.ndarray, truth: np.ndarray) -> float:
    """Exact E_{Y~truth}[S(pmf_row, Y)] = sum_k truth_k * S(pmf_row, k).

    Deterministic (no Monte Carlo): the discrete expectation is a finite sum,
    so propriety assertions below are exact up to floating point.
    """
    k = truth.size
    rows = np.tile(np.asarray(pmf_row, dtype=float).reshape(1, -1), (k, 1))
    per_category = score_fn(rows, np.arange(k, dtype=float))
    return float(np.sum(truth * per_category))


def _candidate_pmfs(truth: np.ndarray, seed: int) -> list[np.ndarray]:
    """Seeded mis-specified candidates: Dirichlet draws + uniform mixtures."""
    rng = np.random.default_rng(seed)
    k = truth.size
    cands = [rng.dirichlet(np.ones(k) * 4.0) for _ in range(12)]
    uniform = np.full(k, 1.0 / k)
    for lam in (0.15, 0.35, 0.55, 0.75, 0.95):
        cands.append((1.0 - lam) * truth + lam * uniform)
    return cands


# ---------------------------------------------------------------------------
# Ranked probability score (Epstein 1969) — equals the discrete CRPS
# ---------------------------------------------------------------------------


def test_rps_hand_computable_wikipedia_example() -> None:
    # Wikipedia "Scoring rule" example: RPS((0.5, 0.5, 0), class 1) = 0.25 and
    # RPS((0.5, 0, 0.5), class 1) = 0.5 (1-indexed classes; here y=0).
    p1 = np.array([[0.5, 0.5, 0.0]])
    p2 = np.array([[0.5, 0.0, 0.5]])
    y = np.array([0.0])
    assert ranked_probability_score(p1, y)[0] == pytest.approx(0.25, abs=1e-12)
    assert ranked_probability_score(p2, y)[0] == pytest.approx(0.5, abs=1e-12)


def test_rps_perfect_forecast_is_zero() -> None:
    pmfs = np.array([[0.0, 0.0, 1.0]])
    assert ranked_probability_score(pmfs, np.array([2.0]))[0] == pytest.approx(0.0, abs=1e-15)


def test_rps_matches_manual_cdf_indicator_formula() -> None:
    # RPS = sum_i (F_i - 1{y <= i})^2 computed by hand on one row.
    rng = np.random.default_rng(101)
    p = rng.dirichlet(np.ones(6))
    y = 3.0
    cdf = np.cumsum(p)
    ind = (np.arange(6, dtype=float) >= y).astype(float)
    expected = float(np.sum((cdf - ind) ** 2))
    got = ranked_probability_score(p.reshape(1, -1), np.array([y]))[0]
    assert got == pytest.approx(expected, abs=1e-12)


def test_rps_penalizes_far_mass_more_than_near_mass() -> None:
    # Ordinal sensitivity: same probability on the wrong side, different
    # distance from the realized category.
    y = np.array([2.0])
    near = np.array([[0.0, 0.5, 0.5]])
    far = np.array([[0.5, 0.5, 0.0]])
    assert ranked_probability_score(near, y)[0] < ranked_probability_score(far, y)[0]


def test_rps_propriety_truth_minimizes_exact_expectation() -> None:
    truth_score = _expected_score_under_truth(ranked_probability_score, TRUTH_K5, TRUTH_K5)
    for cand in _candidate_pmfs(TRUTH_K5, seed=11):
        cand_score = _expected_score_under_truth(ranked_probability_score, cand, TRUTH_K5)
        assert truth_score < cand_score


def test_rps_expected_value_at_truth_closed_form() -> None:
    # E_G[RPS(G, Y)] = sum_i CDF_i (1 - CDF_i) — the Kolassa potential term.
    cdf = np.cumsum(TRUTH_K5)
    expected = float(np.sum(cdf * (1.0 - cdf)))
    got = _expected_score_under_truth(ranked_probability_score, TRUTH_K5, TRUTH_K5)
    assert got == pytest.approx(expected, abs=1e-12)


def test_rps_mc_mean_converges_to_exact_expectation() -> None:
    rng = np.random.default_rng(2024)
    draws = rng.choice(5, size=20000, p=TRUTH_K5).astype(float)
    mc = mean_ranked_probability_score(np.tile(TRUTH_K5, (20000, 1)), draws)
    exact = _expected_score_under_truth(ranked_probability_score, TRUTH_K5, TRUTH_K5)
    assert mc == pytest.approx(exact, abs=0.02)


def test_rps_support_grid_shifts_indicators() -> None:
    # support = (10, 20, 30), y = 15 -> indicators (0, 1, 1).
    p = np.array([[0.2, 0.3, 0.5]])
    got = ranked_probability_score(p, np.array([15.0]), support=np.array([10.0, 20.0, 30.0]))[0]
    expected = (0.2 - 0.0) ** 2 + (0.5 - 1.0) ** 2 + (1.0 - 1.0) ** 2
    assert got == pytest.approx(expected, abs=1e-12)


def test_rps_invalid_pmf_rows_and_nan_y_mask_to_nan() -> None:
    pmfs = np.array(
        [
            [0.5, 0.5],  # valid
            [0.5, 0.4],  # sums to 0.9 -> NaN (fail closed, no renormalization)
            [-0.2, 1.2],  # negative -> NaN
            [0.6, 0.4 + 1e-10],  # within 1e-6 tolerance -> renormalized, finite
        ]
    )
    y = np.array([0.0, 1.0, 0.0, 1.0])
    out = ranked_probability_score(pmfs, y)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isfinite(out[3])
    out_nan_y = ranked_probability_score(np.array([[0.5, 0.5]]), np.array([np.nan]))
    assert np.isnan(out_nan_y[0])


def test_rps_edges_empty_mismatch_and_mean_nan() -> None:
    empty = ranked_probability_score(np.empty((0, 3)), np.empty(0))
    assert empty.shape == (0,)
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([[0.5, 0.5]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([[0.5, 0.5]]), np.array([[0.0]]))
    all_bad = np.array([[0.4, 0.4], [0.3, 0.3]])
    assert np.isnan(mean_ranked_probability_score(all_bad, np.array([0.0, 1.0])))
    assert np.isnan(mean_ranked_probability_score(np.empty((0, 2)), np.empty(0)))


def test_rps_single_pmf_broadcasts_over_observations() -> None:
    y = np.array([0.0, 1.0, 2.0])
    out = ranked_probability_score(TRUTH_K5[:3] / TRUTH_K5[:3].sum(), y)
    assert out.shape == (3,)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Multi-category Brier (Brier 1950; Gneiting-Raftery 2007)
# ---------------------------------------------------------------------------


def test_brier_multiclass_hand_computable() -> None:
    p = np.array([[0.7, 0.2, 0.1]])
    got = brier_multiclass(p, np.array([0.0]))[0]
    expected = (0.7 - 1.0) ** 2 + 0.2**2 + 0.1**2
    assert got == pytest.approx(expected, abs=1e-12)


def test_brier_multiclass_binary_is_double_probability_brier() -> None:
    # K=2 columns (1-p, p): sum_k (p_k - o_k)^2 = 2 (p - y)^2, i.e. exactly
    # twice metrics.probability.brier_score (documented convention).
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.98, size=500)
    y = (rng.random(500) < p).astype(float)
    probs2 = np.column_stack([1.0 - p, p])
    assert mean_brier_multiclass(probs2, y) == pytest.approx(2.0 * brier_score(p, y), abs=1e-12)


def test_brier_multiclass_within_zero_two_range() -> None:
    rng = np.random.default_rng(6)
    pmfs = rng.dirichlet(np.ones(4), size=300)
    y = rng.integers(0, 4, size=300).astype(float)
    out = brier_multiclass(pmfs, y)
    assert np.all(out >= 0.0) and np.all(out <= 2.0)


def test_brier_multiclass_propriety_truth_minimizes_exact_expectation() -> None:
    truth_score = _expected_score_under_truth(brier_multiclass, TRUTH_K5, TRUTH_K5)
    for cand in _candidate_pmfs(TRUTH_K5, seed=12):
        assert truth_score < _expected_score_under_truth(brier_multiclass, cand, TRUTH_K5)


def test_brier_multiclass_expected_value_at_truth_closed_form() -> None:
    # E_G[BS(G, Y)] = sum_k G_k (1 - G_k).
    expected = float(np.sum(TRUTH_K5 * (1.0 - TRUTH_K5)))
    got = _expected_score_under_truth(brier_multiclass, TRUTH_K5, TRUTH_K5)
    assert got == pytest.approx(expected, abs=1e-12)


def test_brier_multiclass_invalid_labels_mask_to_nan() -> None:
    p = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    y = np.array([0.0, 1.0, 2.0, 0.5])  # 2.0 out of range, 0.5 non-integer
    out = brier_multiclass(p, y)
    assert np.isfinite(out[:2]).all()
    assert np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(mean_brier_multiclass(p[2:], y[2:]))


# ---------------------------------------------------------------------------
# Discrete logarithmic score
# ---------------------------------------------------------------------------


def test_log_score_discrete_hand_computable() -> None:
    p = np.array([[0.1, 0.6, 0.3]])
    got = log_score_discrete(p, np.array([2.0]))[0]
    assert got == pytest.approx(-np.log(0.3), abs=1e-12)


def test_log_score_discrete_binary_matches_log_loss() -> None:
    # K=2 columns (1-p, p) reproduce metrics.probability.log_loss (its eps
    # clip is inactive for probabilities bounded away from 0 and 1).
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, size=400)
    y = (rng.random(400) < p).astype(float)
    probs2 = np.column_stack([1.0 - p, p])
    assert mean_log_score_discrete(probs2, y) == pytest.approx(log_loss(p, y), abs=1e-12)


def test_log_score_discrete_propriety_truth_equals_entropy() -> None:
    truth_score = _expected_score_under_truth(log_score_discrete, TRUTH_K5, TRUTH_K5)
    entropy = float(-np.sum(TRUTH_K5 * np.log(TRUTH_K5)))
    assert truth_score == pytest.approx(entropy, abs=1e-12)
    for cand in _candidate_pmfs(TRUTH_K5, seed=13):
        assert truth_score < _expected_score_under_truth(log_score_discrete, cand, TRUTH_K5)


def test_log_score_discrete_zero_probability_is_infinite_not_clipped() -> None:
    # A realized outcome predicted impossible scores +inf — honest, no eps pad.
    p = np.array([[0.0, 1.0]])
    out = log_score_discrete(p, np.array([0.0]))
    assert np.isinf(out[0]) and out[0] > 0.0
    assert np.isinf(mean_log_score_discrete(p, np.array([0.0])))


def test_log_score_discrete_invalid_labels_mask_to_nan() -> None:
    p = np.array([[0.5, 0.5]])
    assert np.isnan(log_score_discrete(p, np.array([-1.0]))[0])
    assert np.isnan(log_score_discrete(p, np.array([1.5]))[0])
    empty = log_score_discrete(np.empty((0, 2)), np.empty(0))
    assert empty.shape == (0,)


# ---------------------------------------------------------------------------
# Spherical score (Good 1952; loss orientation)
# ---------------------------------------------------------------------------


def test_spherical_score_hand_computable() -> None:
    p = np.array([[0.5, 0.5]])
    got = spherical_score(p, np.array([0.0]))[0]
    assert got == pytest.approx(1.0 - 0.5 / np.sqrt(0.5), abs=1e-12)


def test_spherical_score_perfect_forecast_is_zero() -> None:
    p = np.array([[0.0, 1.0, 0.0]])
    assert spherical_score(p, np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-15)


def test_spherical_score_uniform_value_and_range() -> None:
    k = 4
    rng = np.random.default_rng(8)
    pmfs = rng.dirichlet(np.ones(k), size=200)
    y = rng.integers(0, k, size=200).astype(float)
    out = spherical_score(pmfs, y)
    assert np.all(out >= 0.0) and np.all(out < 1.0)
    uniform = np.full((1, k), 1.0 / k)
    got = spherical_score(uniform, np.array([0.0]))[0]
    assert got == pytest.approx(1.0 - 1.0 / np.sqrt(k), abs=1e-12)
    assert mean_spherical_score(pmfs, y) == pytest.approx(float(np.mean(out)), abs=1e-12)
    assert np.isnan(mean_spherical_score(np.empty((0, k)), np.empty(0)))


def test_spherical_score_propriety_truth_minimizes_exact_expectation() -> None:
    truth_score = _expected_score_under_truth(spherical_score, TRUTH_K5, TRUTH_K5)
    # E_G[S(G, Y)] = 1 - ||G||_2 at the truth.
    assert truth_score == pytest.approx(1.0 - float(np.linalg.norm(TRUTH_K5)), abs=1e-12)
    for cand in _candidate_pmfs(TRUTH_K5, seed=14):
        assert truth_score < _expected_score_under_truth(spherical_score, cand, TRUTH_K5)


# ---------------------------------------------------------------------------
# Exact multi-category Brier decomposition (Murphy 1973; Bröcker 2009)
# ---------------------------------------------------------------------------


def _seeded_multiclass_data(
    seed: int, n: int = 300, pool: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.dirichlet(np.ones(4) * 2.0, size=pool)
    picks = rng.integers(0, pool, size=n)
    probs = vectors[picks]
    labels = np.array([rng.choice(4, p=probs[t]) for t in range(n)], dtype=float)
    return probs, labels


def test_brier_decomposition_identity_exact_on_repeated_forecasts() -> None:
    probs, labels = _seeded_multiclass_data(seed=21)
    d = brier_decomposition(probs, labels)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["n_bins"] == pytest.approx(6.0)
    assert d["brier"] == pytest.approx(float(np.mean(brier_multiclass(probs, labels))), abs=1e-12)


def test_brier_decomposition_identity_exact_with_all_distinct_forecasts() -> None:
    # Degenerate binning (every forecast unique) must not break the identity.
    rng = np.random.default_rng(22)
    probs = rng.dirichlet(np.ones(3) * 5.0, size=150)
    labels = rng.integers(0, 3, size=150).astype(float)
    d = brier_decomposition(probs, labels)
    assert d["n_bins"] == pytest.approx(150.0)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)


def test_brier_decomposition_terms_are_nonnegative_with_valid_ranges() -> None:
    probs, labels = _seeded_multiclass_data(seed=23)
    d = brier_decomposition(probs, labels)
    assert d["reliability"] >= 0.0
    assert d["resolution"] >= 0.0
    assert 0.0 <= d["uncertainty"] <= 4.0 / 4.0 + 1e-12  # sum_k o(1-o) <= K/4


def test_brier_decomposition_constant_forecast_has_zero_resolution() -> None:
    # One bin only: RES = sum_k (o_bar_k - o_bar_k)^2 = 0 exactly; BS = REL + UNC.
    rng = np.random.default_rng(24)
    n = 400
    labels = rng.choice(5, size=n, p=TRUTH_K5).astype(float)
    probs = np.tile(TRUTH_K5, (n, 1))
    d = brier_decomposition(probs, labels)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-15)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["brier"] == pytest.approx(d["reliability"] + d["uncertainty"], abs=1e-12)


def test_brier_decomposition_calibrated_bins_have_zero_reliability() -> None:
    # Constructed SYNTHETIC data: within each distinct-forecast bin the
    # outcome frequencies equal the forecast pmf exactly -> REL == 0.
    pa = np.array([0.5, 0.5, 0.0])
    pb = np.array([0.25, 0.25, 0.5])
    probs = np.vstack([np.tile(pa, (4, 1)), np.tile(pb, (4, 1))])
    labels = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0])
    d = brier_decomposition(probs, labels)
    assert d["reliability"] == pytest.approx(0.0, abs=1e-15)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["brier"] == pytest.approx(d["uncertainty"] - d["resolution"], abs=1e-12)


def test_brier_decomposition_binary_scale_is_half_multiclass() -> None:
    rng = np.random.default_rng(25)
    p = np.round(rng.uniform(0.05, 0.95, size=200), 2)
    y = (rng.random(200) < p).astype(float)
    probs2 = np.column_stack([1.0 - p, p])
    multi = brier_decomposition(probs2, y)
    binary = brier_decomposition_binary(p, y)
    for key in ("brier", "reliability", "resolution", "uncertainty"):
        assert multi[key] == pytest.approx(2.0 * binary[key], abs=1e-12)


def test_brier_decomposition_fail_closed_edges() -> None:
    good_y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):  # row sum off by more than 1e-6
        brier_decomposition(np.array([[0.5, 0.4], [0.5, 0.6]]), good_y)
    with pytest.raises(ValueError):  # negative probability
        brier_decomposition(np.array([[1.2, -0.2], [0.5, 0.5]]), good_y)
    with pytest.raises(ValueError):  # non-finite
        brier_decomposition(np.array([[np.nan, 1.0], [0.5, 0.5]]), good_y)
    with pytest.raises(ValueError):  # non-integer label
        brier_decomposition(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0.5, 1.0]))
    with pytest.raises(ValueError):  # out-of-range label
        brier_decomposition(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0.0, 7.0]))
    with pytest.raises(ValueError):  # length mismatch
        brier_decomposition(np.array([[0.5, 0.5]]), good_y)
    with pytest.raises(ValueError):  # empty
        brier_decomposition(np.empty((0, 2)), np.empty(0))


def test_brier_decomposition_is_deterministic() -> None:
    probs, labels = _seeded_multiclass_data(seed=26)
    assert brier_decomposition(probs, labels) == brier_decomposition(probs, labels)


# ---------------------------------------------------------------------------
# Exact binary Brier decomposition — cross-checks against existing modules
# ---------------------------------------------------------------------------


def test_brier_binary_decomposition_hand_example() -> None:
    # f = (0.8, 0.4), o = (1, 0): BS = 0.1, REL = 0.1, RES = 0.25, UNC = 0.25
    # (RES uses bin outcome frequencies: Siegert 2013 eq. 18).
    d = brier_decomposition_binary(np.array([0.8, 0.4]), np.array([1.0, 0.0]))
    assert d["brier"] == pytest.approx(0.1, abs=1e-12)
    assert d["reliability"] == pytest.approx(0.1, abs=1e-12)
    assert d["resolution"] == pytest.approx(0.25, abs=1e-12)
    assert d["uncertainty"] == pytest.approx(0.25, abs=1e-12)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-15)


def test_brier_binary_decomposition_identity_exact_seeded() -> None:
    rng = np.random.default_rng(31)
    p = np.round(rng.uniform(0.02, 0.98, size=250), 2)  # repeated values -> real bins
    y = (rng.random(250) < p).astype(float)
    d = brier_decomposition_binary(p, y)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["reliability"] >= 0.0 and d["resolution"] >= 0.0


def test_brier_binary_matches_calibration2_when_bins_align() -> None:
    # Every distinct forecast value in its own equal-width bin: the exact
    # decomposition coincides with calibration2.murphy_decomposition and that
    # function's binned decomp_error vanishes. Complements (does not replace)
    # the existing approximate implementation.
    rng = np.random.default_rng(32)
    values = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    p = rng.choice(values, size=400)
    y = (rng.random(400) < p).astype(float)
    mine = brier_decomposition_binary(p, y)
    theirs = murphy_decomposition(p, y, n_bins=10)
    for key in ("brier", "reliability", "resolution", "uncertainty"):
        assert mine[key] == pytest.approx(theirs[key], abs=1e-12)
    assert mine["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert theirs["decomp_error"] == pytest.approx(0.0, abs=1e-12)


def test_brier_binary_brier_key_equals_probability_brier_score() -> None:
    rng = np.random.default_rng(33)
    p = rng.uniform(0.0, 1.0, size=120)
    y = (rng.random(120) < p).astype(float)
    d = brier_decomposition_binary(p, y)
    assert d["brier"] == pytest.approx(brier_score(p, y), abs=1e-15)


def test_brier_binary_fail_closed_edges() -> None:
    with pytest.raises(ValueError):  # non-binary outcome
        brier_decomposition_binary(np.array([0.5, 0.5]), np.array([0.0, 2.0]))
    with pytest.raises(ValueError):  # probability out of range
        brier_decomposition_binary(np.array([0.5, 1.5]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):  # non-finite probability
        brier_decomposition_binary(np.array([np.nan, 0.5]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):  # length mismatch
        brier_decomposition_binary(np.array([0.5]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):  # empty
        brier_decomposition_binary(np.empty(0), np.empty(0))


# ---------------------------------------------------------------------------
# Bröcker (2012): ensemble CRPS = potential + quality − reliability
# ---------------------------------------------------------------------------


def test_broecker_identity_exact_and_matches_fair_crps() -> None:
    rng = np.random.default_rng(41)
    ens = rng.normal(size=(40, 8))
    y = rng.normal(size=40)
    d = broecker_ensemble_crps_decomposition(ens, y)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["crps"] == pytest.approx(float(np.mean(fair_ensemble_crps(ens, y))), abs=1e-12)
    # crps == mean_abs_error_term − mean_spread_term (fair CRPS structure)
    assert d["crps"] == pytest.approx(d["mean_abs_error_term"] - d["mean_spread_term"], abs=1e-12)


def test_fair_crps_relates_exactly_to_crps_empirical() -> None:
    # scoring.crps_empirical is the plug-in (1/(2n^2)) estimator; the fair
    # estimator differs by exactly U/(2n) with U the unbiased mean pairwise
    # member distance (dedup cross-check of the reused module).
    rng = np.random.default_rng(42)
    m, n = 25, 6
    ens = rng.normal(size=(m, n))
    y = rng.normal(size=m)
    fair = fair_ensemble_crps(ens, y)
    pair = np.abs(ens[:, :, None] - ens[:, None, :])
    u_stat = pair.sum(axis=(1, 2)) / float(n * (n - 1))
    plugin = np.array([crps_empirical(y[t], ens[t]) for t in range(m)])
    assert np.allclose(plugin, fair + u_stat / (2.0 * n), atol=1e-12)


def test_broecker_perfect_ensemble_approaches_potential() -> None:
    # SYNTHETIC: members and observations iid N(0, 1) (exchangeable). Then
    # quality -> 0, reliability -> 0, and potential -> E|Y-Y'|/2 = 1/sqrt(pi),
    # the population CRPS of a perfect Gaussian forecast.
    rng = np.random.default_rng(777)
    y = rng.normal(size=4000)
    ens = rng.normal(size=(4000, 100))
    d = broecker_ensemble_crps_decomposition(ens, y)
    assert d["potential"] == pytest.approx(1.0 / np.sqrt(np.pi), abs=0.03)
    assert abs(d["quality"]) < 0.05
    assert abs(d["reliability"]) < 0.05
    assert d["crps"] == pytest.approx(d["potential"], abs=0.05)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)


def test_broecker_overdispersed_ensemble_positive_reliability() -> None:
    rng = np.random.default_rng(778)
    y = rng.normal(size=800)
    ens = rng.normal(size=(800, 20)) * 3.0  # spread >> observation spread
    d = broecker_ensemble_crps_decomposition(ens, y)
    assert d["reliability"] > 0.0
    assert d["quality"] > 0.0
    assert d["crps"] > d["potential"]


def test_broecker_underdispersed_ensemble_negative_reliability() -> None:
    rng = np.random.default_rng(779)
    y = rng.normal(size=800)
    ens = rng.normal(size=(800, 20)) * 0.3  # spread << observation spread
    d = broecker_ensemble_crps_decomposition(ens, y)
    assert d["reliability"] < 0.0  # penalizes through -reliability > 0
    assert d["crps"] > d["potential"]


def test_broecker_is_deterministic() -> None:
    rng = np.random.default_rng(43)
    ens = rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    first = broecker_ensemble_crps_decomposition(ens, y)
    second = broecker_ensemble_crps_decomposition(ens, y)
    assert first == second


def test_broecker_fail_closed_edges() -> None:
    ens = np.random.default_rng(44).normal(size=(4, 3))
    y = np.random.default_rng(45).normal(size=4)
    with pytest.raises(ValueError):  # 1d ensembles
        broecker_ensemble_crps_decomposition(ens[0], y)
    with pytest.raises(ValueError):  # fewer than 2 members
        broecker_ensemble_crps_decomposition(ens[:, :1], y)
    with pytest.raises(ValueError):  # fewer than 2 forecasts
        broecker_ensemble_crps_decomposition(ens[:1], y[:1])
    with pytest.raises(ValueError):  # length mismatch
        broecker_ensemble_crps_decomposition(ens, y[:3])
    bad = np.array(ens)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):  # non-finite ensemble
        broecker_ensemble_crps_decomposition(bad, y)
    bad_y = np.array(y)
    bad_y[1] = np.inf
    with pytest.raises(ValueError):  # non-finite observation
        broecker_ensemble_crps_decomposition(ens, bad_y)


def test_fair_ensemble_crps_masks_bad_rows() -> None:
    rng = np.random.default_rng(46)
    ens = rng.normal(size=(5, 4))
    ens[2, 1] = np.nan
    y = rng.normal(size=5)
    out = fair_ensemble_crps(ens, y)
    assert np.isnan(out[2])
    assert np.isfinite(out[[0, 1, 3, 4]]).all()
    y_nan = np.array(y)
    y_nan[0] = np.nan
    assert np.isnan(fair_ensemble_crps(ens, y_nan)[0])
    empty = fair_ensemble_crps(np.empty((0, 3)), np.empty(0))
    assert empty.shape == (0,)


# ---------------------------------------------------------------------------
# Kolassa (2016): discrete/count CRPS decomposition
# ---------------------------------------------------------------------------


def test_kolassa_fixed_pmf_two_term_form() -> None:
    # Single fixed forecast: one bin per threshold -> resolution == 0 exactly
    # and reliability == deviation == sum_i (F_i - Ghat_i)^2, recovering
    # Kolassa's two-term mean CRPS = deviation + potential.
    rng = np.random.default_rng(51)
    y = rng.poisson(2.5, size=600).astype(float)
    pmf = poisson.pmf(np.arange(12), 2.5)
    pmf = pmf / pmf.sum()
    d = kolassa_crps_decomposition(pmf, y)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-15)
    assert d["reliability"] == pytest.approx(d["deviation"], abs=1e-14)
    assert d["crps"] == pytest.approx(d["deviation"] + d["potential"], abs=1e-12)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)


def test_kolassa_identity_exact_on_varying_forecasts() -> None:
    rng = np.random.default_rng(52)
    k = 7
    n = 500
    pmfs = rng.dirichlet(np.ones(k) * 3.0, size=n)
    y = rng.integers(0, k, size=n).astype(float)
    d = kolassa_crps_decomposition(pmfs, y)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["crps"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["potential"], abs=1e-12
    )
    assert d["potential"] >= 0.0
    assert d["reliability"] >= 0.0 and d["resolution"] >= 0.0


def test_kolassa_crps_key_matches_mean_rps() -> None:
    rng = np.random.default_rng(53)
    k = 6
    n = 250
    pmfs = rng.dirichlet(np.ones(k) * 2.0, size=n)
    y = rng.integers(0, k, size=n).astype(float)
    d = kolassa_crps_decomposition(pmfs, y)
    assert d["crps"] == pytest.approx(mean_ranked_probability_score(pmfs, y), abs=1e-12)


def test_kolassa_potential_is_forecast_free_and_attained_at_empirical_pmf() -> None:
    # Forecasting the empirical pmf of the same sample: crps == potential
    # exactly (reliability/deviation vanish, resolution is zero) — the
    # discrete analogue of Bröcker's perfect-forecast floor.
    rng = np.random.default_rng(54)
    y = rng.poisson(3.0, size=800).astype(float)
    k = int(y.max()) + 1
    emp = np.bincount(y.astype(int), minlength=k).astype(float)
    emp /= emp.sum()
    d = kolassa_crps_decomposition(emp, y)
    assert d["crps"] == pytest.approx(d["potential"], abs=1e-12)
    assert d["reliability"] < 1e-20
    assert d["resolution"] == pytest.approx(0.0, abs=1e-15)
    assert d["deviation"] < 1e-20
    # potential == sum_i Ghat_i (1 - Ghat_i) computed independently
    g_hat = np.mean(y[:, None] <= np.arange(k, dtype=float)[None, :], axis=0)
    assert d["potential"] == pytest.approx(float(np.sum(g_hat * (1.0 - g_hat))), abs=1e-12)


def test_kolassa_poisson_truth_beats_misspecified_mc() -> None:
    # SYNTHETIC propriety in the count setting: the (truncated, renormalized)
    # true Poisson(3) pmf scores lower than a mis-specified Poisson(6).
    k = 25
    grid = np.arange(k)
    true_pmf = poisson.pmf(grid, 3.0)
    true_pmf = true_pmf / true_pmf.sum()
    mis_pmf = poisson.pmf(grid, 6.0)
    mis_pmf = mis_pmf / mis_pmf.sum()
    for seed in (11, 12, 13):
        rng = np.random.default_rng(seed)
        y = rng.poisson(3.0, size=4000).astype(float)
        crps_true = mean_ranked_probability_score(np.tile(true_pmf, (4000, 1)), y)
        crps_mis = mean_ranked_probability_score(np.tile(mis_pmf, (4000, 1)), y)
        assert crps_true < crps_mis - 0.1, f"seed {seed}"
        d = kolassa_crps_decomposition(true_pmf, y)
        assert d["crps"] == pytest.approx(crps_true, abs=1e-12)
        # near-true forecast: almost all score mass is intrinsic potential
        assert d["potential"] < d["crps"]
        assert d["deviation"] < 0.01


def test_kolassa_accepts_single_row_and_broadcasts() -> None:
    rng = np.random.default_rng(55)
    y = rng.poisson(1.5, size=200).astype(float)
    pmf = poisson.pmf(np.arange(10), 1.5)
    pmf = pmf / pmf.sum()
    d_1d = kolassa_crps_decomposition(pmf, y)
    d_2d = kolassa_crps_decomposition(np.tile(pmf, (200, 1)), y)
    assert d_1d == d_2d


def test_kolassa_fail_closed_edges() -> None:
    y = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):  # row sum off
        kolassa_crps_decomposition(np.array([[0.5, 0.4, 0.0]]), y)
    with pytest.raises(ValueError):  # negative pmf
        kolassa_crps_decomposition(np.array([[1.2, -0.1, -0.1]]), y)
    with pytest.raises(ValueError):  # non-finite y
        kolassa_crps_decomposition(np.array([[0.5, 0.3, 0.2]]), np.array([0.0, np.nan, 2.0]))
    with pytest.raises(ValueError):  # length mismatch
        kolassa_crps_decomposition(np.array([[0.5, 0.5, 0.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):  # empty
        kolassa_crps_decomposition(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError):  # support not strictly increasing
        kolassa_crps_decomposition(
            np.array([[0.5, 0.3, 0.2]]), y, support=np.array([0.0, 0.0, 1.0])
        )
    with pytest.raises(ValueError):  # support length mismatch
        kolassa_crps_decomposition(np.array([[0.5, 0.3, 0.2]]), y, support=np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# sharpness_decomposition (Gneiting-type: spread vs mean-error contribution)
# ---------------------------------------------------------------------------


def test_sharpness_identity_and_reuse_of_scoring_module() -> None:
    rng = np.random.default_rng(61)
    y = rng.normal(size=300)
    mu = rng.normal(size=300)
    sigma = np.abs(rng.normal(size=300)) + 0.5
    d = sharpness_decomposition(y, mu, sigma)
    # crps is computed by the reused scoring.mean_crps_gaussian, exactly.
    assert d["crps"] == mean_crps_gaussian(y, mu, sigma)
    assert d["decomp_error"] == pytest.approx(0.0, abs=1e-12)
    assert d["error_contribution"] >= 0.0
    assert d["sharpness"] > 0.0
    assert d["n"] == pytest.approx(300.0)


def test_sharpness_error_contribution_zero_at_median() -> None:
    mu = np.array([0.0, 1.5, -2.0])
    sigma = np.array([1.0, 0.5, 2.0])
    d = sharpness_decomposition(mu, mu, sigma)  # y == mu -> z == 0
    assert d["error_contribution"] == pytest.approx(0.0, abs=1e-12)
    assert d["crps"] == pytest.approx(d["sharpness"], abs=1e-12)
    # hand value: sigma * (sqrt(2/pi) - 1/sqrt(pi))
    expected = float(np.mean(sigma * (np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))))
    assert d["sharpness"] == pytest.approx(expected, abs=1e-12)


def test_sharpness_elementwise_matches_crps_gaussian() -> None:
    rng = np.random.default_rng(62)
    y = rng.normal(size=50)
    mu = rng.normal(size=50)
    sigma = np.abs(rng.normal(size=50)) + 0.25
    d = sharpness_decomposition(y, mu, sigma)
    elementwise_crps = crps_gaussian(y, mu, sigma)
    # sharpness + error == elementwise CRPS in the mean, and the split is
    # monotone in sigma: doubling sigma doubles the sharpness term.
    d2 = sharpness_decomposition(y, mu, 2.0 * sigma)
    assert d2["sharpness"] == pytest.approx(2.0 * d["sharpness"], abs=1e-12)
    assert d["crps"] == pytest.approx(float(np.mean(elementwise_crps)), abs=1e-12)


def test_sharpness_fail_closed_and_nan_edges() -> None:
    with pytest.raises(ValueError):  # length mismatch
        sharpness_decomposition(np.zeros(3), np.zeros(2), np.ones(3))
    empty = sharpness_decomposition(np.empty(0), np.empty(0), np.empty(0))
    assert np.isnan(empty["sharpness"]) and np.isnan(empty["crps"])
    assert empty["n"] == 0.0
    # sigma <= 0 / non-finite -> masked exactly like crps_gaussian (n counts
    # only valid entries); all-invalid -> honest NaN dict.
    y = np.array([0.0, 1.0, 2.0])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([-1.0, 0.0, np.nan])
    bad = sharpness_decomposition(y, mu, sigma)
    assert np.isnan(bad["crps"]) and bad["n"] == 0.0
    partial = sharpness_decomposition(y, mu, np.array([1.0, -0.5, 2.0]))
    assert partial["n"] == pytest.approx(2.0)
    assert partial["crps"] == mean_crps_gaussian(y, mu, np.array([1.0, -0.5, 2.0]))


# ---------------------------------------------------------------------------
# Honesty contract wiring
# ---------------------------------------------------------------------------


def test_decomposition_dicts_carry_no_forbidden_research_metric_keys() -> None:
    # AGENTS.md honesty contract #1: proper-score keys only, never headline
    # Sharpe/Sortino/Calmar/P&L/NAV tokens (checked with the lab's own guard).
    rng = np.random.default_rng(71)
    probs = rng.dirichlet(np.ones(3), size=20)
    labels = rng.integers(0, 3, size=20).astype(float)
    p_bin = np.round(rng.uniform(0.1, 0.9, size=20), 2)
    y_bin = (rng.random(20) < p_bin).astype(float)
    ens = rng.normal(size=(10, 4))
    y_ens = rng.normal(size=10)
    blob = {
        "brier": brier_decomposition(probs, labels),
        "brier_binary": brier_decomposition_binary(p_bin, y_bin),
        "broecker": broecker_ensemble_crps_decomposition(ens, y_ens),
        "kolassa": kolassa_crps_decomposition(probs, labels),
        "sharpness": sharpness_decomposition(y_ens, y_ens, np.abs(y_ens) + 1.0),
    }
    assert family_blob_forbidden_metrics_absent(blob)
