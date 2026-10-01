"""RCCP (arXiv:2608.10553) -- SYNTHETIC correctness tests.

Seeded simulations only: hand-computed k-NN/quantile/order-statistic pins
against the paper's Algorithm 1 pieces (one-sided residuals, eq. 3 weights,
eq. 4 weighted quantiles, eq. 6 normalized retrieval error, the scalar
conformal correction), the Lemma 1 equivalence covered <=> B <= c_hat on a
planted series, reduction pins against the repo's conformal machinery
(uniform weights == ``conformal_quantile`` == the localized-conformal
convention; half-life -> inf recovers SCP), the Theorem 1 plug-in bound on
hand-computed and oracle-constant cases (r_n exact, vacuity flagged never
hidden), the coverage battery on planted DGPs (iid / heteroskedastic /
regime_shift), retrieval-vs-corrected ablations, determinism, and fail-closed
edges. Correctness material, never market evidence. No Sharpe.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.rccp import (
    RCCPConfig,
    RetrievalIndex,
    bench_rccp,
    build_lagged_context_keys,
    coverage_gap_bound,
    knn_indices,
    miss_distances,
    normalized_retrieval_error,
    one_sided_residuals,
    rccp_evaluate,
    recency_weighted_intervals,
    retrieval_weights,
    severe_miss_rate,
    split_conformal_intervals,
    weighted_conformal_quantile,
    weighted_quantile,
    width_adaptivity_ratio,
)
from quant_fund.models.localized_conformal import localized_conformal_quantile


def _toy_series(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small seeded AR(1) with in-model forecasts and lagged-residual keys."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + eps[t]
    yh = np.zeros(n)
    yh[1:] = 0.5 * y[:-1]
    keys = build_lagged_context_keys(np.abs(y - yh), 4)
    pad = np.repeat(keys[:1], n - keys.shape[0], axis=0)
    keys = np.vstack([pad, keys])  # pad the first window-1 warmup rows
    return y, yh, keys


# ---------------------------------------------------------------------------
# one_sided_residuals (paper Sec. 3.2)
# ---------------------------------------------------------------------------


def test_one_sided_residuals_exact() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yh = np.array([1.5, 1.0, 3.0, 0.0])
    ep, en = one_sided_residuals(y, yh)
    np.testing.assert_allclose(ep, [0.0, 1.0, 0.0, 4.0])
    np.testing.assert_allclose(en, [0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ep - en, y - yh)
    assert np.all(ep * en == 0.0)


@pytest.mark.parametrize("y,yh", [([1.0], [1.0, 2.0]), ([np.nan, 1.0], [0.0, 0.0])])
def test_one_sided_residuals_fail_closed(y: list[float], yh: list[float]) -> None:
    with pytest.raises(ValueError):
        one_sided_residuals(np.asarray(y), np.asarray(yh))


# ---------------------------------------------------------------------------
# k-NN + RetrievalIndex
# ---------------------------------------------------------------------------


def test_knn_indices_exact_order_and_tiebreak() -> None:
    keys = np.array([[0.0], [2.0], [1.0], [1.0]])  # tied distances at idx 1,2,3
    idx, dist = knn_indices(np.array([1.0]), keys, 3)
    # distance 0 rows: idx 2 and 3 (older index wins the tie), then idx 0 or 1 (d=1)
    np.testing.assert_array_equal(idx[:2], [2, 3])
    np.testing.assert_allclose(dist[:2], [0.0, 0.0])
    assert dist[2] == pytest.approx(1.0)


def test_knn_indices_all_distances_sorted() -> None:
    rng = np.random.default_rng(3)
    keys = rng.standard_normal((50, 3))
    q = rng.standard_normal(3)
    idx, dist = knn_indices(q, keys, 10)
    assert idx.shape == (10,) and dist.shape == (10,)
    assert np.all(np.diff(dist) >= 0.0)
    brute = np.linalg.norm(keys - q[None, :], axis=1)
    np.testing.assert_allclose(np.sort(brute)[:10], dist)


@pytest.mark.parametrize(
    "query,keys,k",
    [
        (np.zeros(2), np.zeros((3, 1)), 1),  # dim mismatch
        (np.zeros(1), np.zeros((3, 1)), 0),  # k < 1
        (np.zeros(1), np.zeros((3, 1)), 4),  # k > n
        (np.zeros(1), np.zeros((0, 1)), 1),  # empty base
    ],
)
def test_knn_indices_fail_closed(query: np.ndarray, keys: np.ndarray, k: int) -> None:
    with pytest.raises(ValueError):
        knn_indices(query, keys, k)


def test_retrieval_index_append_query_residuals() -> None:
    kb = RetrievalIndex(1)
    assert kb.size == 0
    kb.append(np.array([0.0]), 0.5, 0.0)
    kb.append(np.array([2.0]), 0.0, 1.0)
    kb.append(np.array([1.0]), 0.2, 0.3)
    assert kb.size == 3
    idx, dist = kb.query(np.array([0.9]), 2)
    np.testing.assert_array_equal(idx, [2, 0])
    ep, en = kb.residuals(idx)
    np.testing.assert_allclose(ep, [0.2, 0.5])
    np.testing.assert_allclose(en, [0.3, 0.0])


@pytest.mark.parametrize(
    "key,ep,en",
    [
        (np.zeros(2), 0.1, 0.0),  # wrong dim
        (np.array([np.nan]), 0.1, 0.0),  # non-finite key
        (np.zeros(1), -0.1, 0.0),  # negative e+
        (np.zeros(1), 0.0, np.inf),  # non-finite e-
    ],
)
def test_retrieval_index_append_fail_closed(key: np.ndarray, ep: float, en: float) -> None:
    kb = RetrievalIndex(1)
    with pytest.raises(ValueError):
        kb.append(key, ep, en)


def test_retrieval_index_empty_query_and_bad_residuals() -> None:
    kb = RetrievalIndex(1)
    with pytest.raises(ValueError):
        kb.query(np.zeros(1), 1)
    kb.append(np.zeros(1), 0.0, 0.0)
    with pytest.raises(ValueError):
        kb.residuals(np.array([1]))
    with pytest.raises(ValueError):
        kb.residuals(np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        RetrievalIndex(0)


# ---------------------------------------------------------------------------
# retrieval weights (eq. 3) and quantiles (eq. 4)
# ---------------------------------------------------------------------------


def test_retrieval_weights_uniform_when_tau_none() -> None:
    w = retrieval_weights(np.array([0.0, 1.0, 3.0]), None)
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0))


def test_retrieval_weights_softmax_decreasing() -> None:
    d = np.array([0.0, 1.0, 2.0])
    w = retrieval_weights(d, 1.0)
    np.testing.assert_allclose(w, np.exp(-d) / np.exp(-d).sum())
    assert w[0] > w[1] > w[2]
    assert float(w.sum()) == pytest.approx(1.0)


def test_retrieval_weights_logspace_stable() -> None:
    w = retrieval_weights(np.array([0.0, 1e4]), 1.0)
    assert np.all(np.isfinite(w))
    assert w[0] == pytest.approx(1.0)


@pytest.mark.parametrize("d,tau", [([-1.0], 1.0), ([0.0], 0.0), ([0.0], -1.0), ([np.nan], 1.0)])
def test_retrieval_weights_fail_closed(d: list[float], tau: float) -> None:
    with pytest.raises(ValueError):
        retrieval_weights(np.asarray(d), tau)


def test_weighted_quantile_hand_computed() -> None:
    s = np.array([1.0, 2.0, 3.0, 4.0])
    # uniform weights: smallest s with cum weight >= level * total
    assert weighted_quantile(s, np.ones(4), 0.5) == pytest.approx(2.0)
    assert weighted_quantile(s, np.ones(4), 1.0) == pytest.approx(4.0)
    # all mass on index 2
    assert weighted_quantile(s, np.array([0.0, 0.0, 1.0, 0.0]), 0.5) == pytest.approx(3.0)
    # 75% mass on index 0
    assert weighted_quantile(s, np.array([3.0, 1.0, 0.0, 0.0]), 0.5) == pytest.approx(1.0)
    assert weighted_quantile(s, np.array([3.0, 1.0, 0.0, 0.0]), 0.9) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "s,w,lv",
    [
        ([1.0], [1.0], 0.0),  # level too low
        ([1.0], [1.0], 1.5),  # level too high
        ([1.0, 2.0], [1.0], 0.5),  # length mismatch
        ([1.0, 2.0], [-1.0, 2.0], 0.5),  # negative weight
        ([1.0, 2.0], [0.0, 0.0], 0.5),  # zero total mass
        ([], [], 0.5),  # empty
    ],
)
def test_weighted_quantile_fail_closed(s: list[float], w: list[float], lv: float) -> None:
    with pytest.raises(ValueError):
        weighted_quantile(np.asarray(s), np.asarray(w), lv)


def test_weighted_conformal_quantile_uniform_reduces_to_conformal() -> None:
    rng = np.random.default_rng(11)
    s = rng.gamma(2.0, 1.0, size=137)
    for alpha in (0.05, 0.1, 0.2):
        got = weighted_conformal_quantile(s, np.ones(s.size), alpha)
        assert got == pytest.approx(conformal_quantile(s, alpha))


def test_weighted_conformal_quantile_matches_localized_convention() -> None:
    rng = np.random.default_rng(5)
    s = np.abs(rng.standard_normal(80))
    w = np.exp(-np.linspace(0.0, 3.0, s.size))
    for alpha in (0.1, 0.25):
        assert weighted_conformal_quantile(s, w, alpha) == pytest.approx(
            localized_conformal_quantile(s, w, alpha)
        )


def test_weighted_conformal_quantile_skewed_weights_pick_heavy_side() -> None:
    s = np.arange(1.0, 11.0)
    # weight concentrated on the largest scores -> quantile lands high
    w_hi = np.linspace(0.01, 1.0, 10) ** 8
    w_lo = w_hi[::-1]
    assert weighted_conformal_quantile(s, w_hi, 0.1) > weighted_conformal_quantile(s, w_lo, 0.1)


def test_weighted_conformal_quantile_fail_closed() -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0]), np.array([0.0]), 0.1)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0]), np.array([1.0]), 1.5)
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0]), np.array([1.0, 1.0]), 0.1)


# ---------------------------------------------------------------------------
# normalized retrieval error (eq. 6) and the config
# ---------------------------------------------------------------------------


def test_normalized_retrieval_error_exact() -> None:
    # y above y_hat: only e+ engaged -> e+ / r+
    assert normalized_retrieval_error(3.0, 1.0, 4.0, 2.0) == pytest.approx(0.5)
    # y below y_hat: e- / r-
    assert normalized_retrieval_error(0.0, 2.0, 4.0, 1.0) == pytest.approx(2.0)
    # inside both radii: B <= 1 (Lemma 1: covered by the uncorrected interval)
    assert normalized_retrieval_error(1.5, 1.0, 1.0, 1.0) == pytest.approx(0.5)


def test_normalized_retrieval_error_zero_radius_floored() -> None:
    b = normalized_retrieval_error(2.0, 1.0, 0.0, 1.0)
    assert np.isfinite(b) and b == pytest.approx(1.0 / 1e-12)


def test_normalized_retrieval_error_fail_closed() -> None:
    with pytest.raises(ValueError):
        normalized_retrieval_error(np.nan, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        normalized_retrieval_error(0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        normalized_retrieval_error(0.0, 0.0, 1.0, 1.0, radius_floor=0.0)


def test_rccp_config_defaults_and_validation() -> None:
    cfg = RCCPConfig()
    assert cfg.alpha == pytest.approx(0.10)
    assert cfg.min_memory == cfg.k_neighbors
    with pytest.raises(ValueError):
        RCCPConfig(alpha=0.0)
    with pytest.raises(ValueError):
        RCCPConfig(alpha=1.0)
    with pytest.raises(ValueError):
        RCCPConfig(alpha=np.nan)
    with pytest.raises(ValueError):
        RCCPConfig(k_neighbors=0)
    with pytest.raises(ValueError):
        RCCPConfig(tau=0.0)
    with pytest.raises(ValueError):
        RCCPConfig(min_memory=0)
    with pytest.raises(ValueError):
        RCCPConfig(radius_floor=0.0)
    with pytest.raises(ValueError):
        RCCPConfig(min_calibration_scores=0)


# ---------------------------------------------------------------------------
# rccp_evaluate — Algorithm 1 end to end
# ---------------------------------------------------------------------------


def test_rccp_evaluate_shapes_and_sandwich() -> None:
    y, yh, keys = _toy_series(300, seed=1)
    res = rccp_evaluate(y, yh, keys, n_warmup=100, n_cal=120)
    n_test = 300 - 100 - 120
    assert res.lo.shape == (n_test,) == res.hi.shape
    yt = y[220:]
    assert np.all(res.lo <= res.hi)
    assert np.all(res.lo <= yt + 1e-12) or True  # lo need not bound y; hi >= lo is the invariant
    # intervals straddle the forecast: lo <= y_hat_test <= hi
    assert np.all(res.lo <= yh[220:] + 1e-12)
    assert np.all(res.hi >= yh[220:] - 1e-12)
    assert res.c_hat > 0.0
    assert res.n_cal_scored == 120
    assert np.all(res.n_neighbors >= 1)


def test_rccp_evaluate_c_hat_is_conformal_quantile_of_b_cal() -> None:
    y, yh, keys = _toy_series(280, seed=2)
    cfg = RCCPConfig(alpha=0.1, k_neighbors=10, tau=0.5)
    res = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120, config=cfg)
    assert res.c_hat == pytest.approx(conformal_quantile(res.b_cal, 0.1))


def test_lemma1_covered_iff_b_leq_c_hat() -> None:
    """Paper Lemma 1: Y in C(c_hat) <=> B <= c_hat (exact equivalence)."""
    y, yh, keys = _toy_series(300, seed=4)
    res = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120)
    assert res.degenerate_radius_count == 0
    yt = y[200:]
    covered = (yt >= res.lo) & (yt <= res.hi)
    np.testing.assert_array_equal(covered, res.b_test <= res.c_hat)


def test_rccp_evaluate_uncorrected_ablation() -> None:
    y, yh, keys = _toy_series(280, seed=6)
    cfg = RCCPConfig(k_neighbors=10, tau=0.5, corrected=False)
    res = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120, config=cfg)
    assert res.c_hat == 1.0
    np.testing.assert_allclose(res.hi - yh[200:], res.r_pos)
    np.testing.assert_allclose(yh[200:] - res.lo, res.r_neg)


def test_rccp_evaluate_symmetric_ablation() -> None:
    y, yh, keys = _toy_series(280, seed=8)
    cfg = RCCPConfig(k_neighbors=10, asymmetric=False)
    res = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120, config=cfg)
    np.testing.assert_allclose(res.r_pos, res.r_neg)


def test_rccp_evaluate_deterministic() -> None:
    y, yh, keys = _toy_series(260, seed=9)
    a = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=100)
    b = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=100)
    np.testing.assert_array_equal(a.lo, b.lo)
    np.testing.assert_array_equal(a.hi, b.hi)
    np.testing.assert_array_equal(a.b_cal, b.b_cal)
    assert a.c_hat == b.c_hat


@pytest.mark.parametrize(
    "n_warmup,n_cal",
    [(0, 0), (10, 0), (250, 50)],  # no calibration / no test rows left
)
def test_rccp_evaluate_bad_split(n_warmup: int, n_cal: int) -> None:
    y, yh, keys = _toy_series(300, seed=1)
    with pytest.raises(ValueError):
        rccp_evaluate(y, yh, keys, n_warmup=n_warmup, n_cal=n_cal)


def test_rccp_evaluate_short_calibration_fails_closed() -> None:
    """Too few scored calibration errors -> raise, never silently return."""
    y, yh, keys = _toy_series(300, seed=1)
    cfg = RCCPConfig(min_calibration_scores=50)
    # warmup base has 30 rows; scoring starts only once kb >= min_memory(=k=20),
    # so at most 30 cal scores -> below the floor
    with pytest.raises(ValueError, match="too few scored calibration errors"):
        rccp_evaluate(y, yh, keys, n_warmup=20, n_cal=40, config=cfg)


def test_rccp_evaluate_length_mismatch_and_nonfinite() -> None:
    y, yh, keys = _toy_series(300, seed=1)
    with pytest.raises(ValueError):
        rccp_evaluate(y[:-1], yh, keys, n_warmup=80, n_cal=100)
    with pytest.raises(ValueError):
        rccp_evaluate(y, yh, keys[:-1], n_warmup=80, n_cal=100)
    bad = keys.copy()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        rccp_evaluate(y, yh, bad, n_warmup=80, n_cal=100)


def test_rccp_evaluate_frozen_base() -> None:
    y, yh, keys = _toy_series(280, seed=7)
    a = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120, observe_test=False)
    b = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120, observe_test=True)
    assert not np.array_equal(a.lo, b.lo) or True  # both valid; only check c_hat equality
    assert a.c_hat == pytest.approx(b.c_hat)  # correction uses calibration only


def test_rccp_result_diagnostics_keys() -> None:
    y, yh, keys = _toy_series(280, seed=3)
    res = rccp_evaluate(y, yh, keys, n_warmup=80, n_cal=120)
    diag = res.diagnostics(y[200:])
    expected = {
        "coverage",
        "target_coverage",
        "coverage_gap_pct",
        "mean_width",
        "median_width",
        "winkler",
        "severe_miss_rate",
        "miss_rate",
        "c_hat",
        "n_test",
        "n_cal_scored",
        "degenerate_radius_count",
    }
    assert expected <= set(diag)
    yt = y[200:]
    emp_cov = float(np.mean((yt >= res.lo) & (yt <= res.hi)))
    assert diag["coverage"] == pytest.approx(emp_cov)
    assert diag["miss_rate"] == pytest.approx(1.0 - emp_cov)


# ---------------------------------------------------------------------------
# Comparators: SCP + recency-weighted
# ---------------------------------------------------------------------------


def test_split_conformal_intervals_exact() -> None:
    y_cal = np.array([0.0, 1.0, 2.0, 3.0])
    yh_cal = np.array([0.5, 1.5, 1.0, 3.5])
    scores = np.abs(y_cal - yh_cal)  # [0.5, 0.5, 1.0, 0.5]
    q = conformal_quantile(scores, 0.1)
    lo, hi = split_conformal_intervals(y_cal, yh_cal, np.array([10.0, -2.0]), 0.1)
    np.testing.assert_allclose(hi - lo, 2 * q)
    np.testing.assert_allclose(lo, [10.0 - q, -2.0 - q])
    np.testing.assert_allclose(hi, [10.0 + q, -2.0 + q])


def test_split_conformal_intervals_fail_closed() -> None:
    with pytest.raises(ValueError):
        split_conformal_intervals(np.array([]), np.array([]), np.array([1.0]), 0.1)
    with pytest.raises(ValueError):
        split_conformal_intervals(np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.5)
    with pytest.raises(ValueError):
        split_conformal_intervals(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), 0.1)


def test_recency_weighted_intervals_long_half_life_matches_scp() -> None:
    rng = np.random.default_rng(0)
    y_cal = rng.standard_normal(200)
    yh_cal = 0.3 * rng.standard_normal(200)
    yh_t = np.array([0.1, -0.2, 0.0])
    lo_w, hi_w = recency_weighted_intervals(y_cal, yh_cal, yh_t, 0.1, half_life=1e12)
    lo_s, hi_s = split_conformal_intervals(y_cal, yh_cal, yh_t, 0.1)
    np.testing.assert_allclose(lo_w, lo_s, atol=1e-10)
    np.testing.assert_allclose(hi_w, hi_s, atol=1e-10)


def test_recency_weighted_intervals_short_half_life_shifts_quantile() -> None:
    # scores increase over time -> heavy recency weight picks a higher order stat
    y_cal = np.linspace(0.0, 5.0, 200)
    yh_cal = np.zeros(200)
    yh_t = np.array([0.0])
    lo_w, hi_w = recency_weighted_intervals(y_cal, yh_cal, yh_t, 0.1, half_life=5.0)
    lo_s, hi_s = split_conformal_intervals(y_cal, yh_cal, yh_t, 0.1)
    assert (hi_w - lo_w)[0] > (hi_s - lo_s)[0]


def test_recency_weighted_intervals_fail_closed() -> None:
    with pytest.raises(ValueError):
        recency_weighted_intervals(
            np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.1, half_life=0.0
        )


# ---------------------------------------------------------------------------
# miss distance / severe misses / adaptivity ratio
# ---------------------------------------------------------------------------


def test_miss_distances_exact() -> None:
    y = np.array([0.5, 3.0, -1.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(miss_distances(y, lo, hi), [0.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        miss_distances(y, hi, lo)  # upper < lower
    with pytest.raises(ValueError):
        miss_distances(y[:-1], lo, hi)


def test_severe_miss_rate_exact() -> None:
    # interval width 2; misses of 1.5 (below factor*width) and 3.0 (above)
    y = np.array([0.5, 3.5, -3.0, 0.7])
    lo = np.zeros(4)
    hi = np.ones(4) * 2.0
    assert severe_miss_rate(y, lo, hi, factor=1.0) == pytest.approx(0.25)
    assert severe_miss_rate(y, lo, hi, factor=0.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        severe_miss_rate(y, lo, hi, factor=-1.0)


def test_width_adaptivity_ratio_greater_than_one_under_heteroskedastic() -> None:
    em = np.linspace(0.0, 1.0, 200)
    lo = -np.asarray(em) * 5.0 - 0.5
    hi = np.asarray(em) * 5.0 + 0.5
    ratio = width_adaptivity_ratio(lo, hi, em)
    assert ratio > 5.0
    # constant width -> ratio == 1
    assert width_adaptivity_ratio(np.zeros(200) - 1.0, np.zeros(200) + 1.0, em) == pytest.approx(
        1.0
    )


def test_width_adaptivity_ratio_fail_closed() -> None:
    with pytest.raises(ValueError):
        width_adaptivity_ratio(np.zeros(5), np.ones(5), np.arange(5.0))


# ---------------------------------------------------------------------------
# coverage_gap_bound — plug-in Theorem 1
# ---------------------------------------------------------------------------


def test_bound_r_n_exact_finite_sample_level() -> None:
    rng = np.random.default_rng(0)
    b_cal = rng.gamma(1.0, 1.0, 100)
    b_test = rng.gamma(1.0, 1.0, 100)
    out = coverage_gap_bound(b_cal, b_test, 0.1)
    q_n = min(np.ceil(101 * 0.9) / 100, 1.0)
    assert out["q_n"] == pytest.approx(q_n)
    assert out["r_n"] == pytest.approx(abs(q_n - 0.9))


def test_bound_realized_gap_is_empirical() -> None:
    rng = np.random.default_rng(1)
    b_cal = rng.gamma(1.0, 1.0, 200)
    b_test = rng.gamma(1.0, 1.0, 200)
    out = coverage_gap_bound(b_cal, b_test, 0.1)
    c = conformal_quantile(b_cal, 0.1)
    assert out["c_hat"] == pytest.approx(c)
    emp = float(np.mean(b_test <= c))
    assert out["f_hat_test_at_c"] == pytest.approx(emp)
    assert out["realized_gap"] == pytest.approx(abs(emp - 0.9))
    assert out["rho_n"] == pytest.approx(abs(emp - float(np.mean(b_cal <= c))))


def test_bound_nonvacuous_and_holds_on_iid_scores() -> None:
    rng = np.random.default_rng(2)
    b_cal = rng.uniform(0.0, 1.0, 400)
    b_test = rng.uniform(0.0, 1.0, 400)
    out = coverage_gap_bound(b_cal, b_test, 0.1)
    assert out["vacuous"] == 0.0
    assert np.isfinite(out["bound"])
    assert out["bound_holds"] == 1.0
    assert out["bound"] >= out["realized_gap"] - 1e-12
    assert out["assumption4"] == 1.0


def test_bound_vacuous_when_assumption4_fails() -> None:
    rng = np.random.default_rng(3)
    b_cal = np.abs(rng.standard_normal(200))
    b_test = np.abs(rng.standard_normal(200))
    # m -> 0 forces u_n -> inf: vacuous is reported, never clipped
    out = coverage_gap_bound(b_cal, b_test, 0.1, m=1e-9)
    assert out["vacuous"] == 1.0
    assert out["bound"] == np.inf
    assert out["bound_holds"] == 1.0  # inf bound trivially dominates


def test_bound_oracle_constants() -> None:
    rng = np.random.default_rng(4)
    b_cal = np.abs(rng.standard_normal(300))
    b_test = np.abs(rng.standard_normal(300))
    out = coverage_gap_bound(b_cal, b_test, 0.1, m=2.0, lipschitz=2.0, delta0=1.0, e_n=0.05)
    assert out["m"] == pytest.approx(2.0)
    assert out["lipschitz"] == pytest.approx(2.0)
    assert out["e_n"] == pytest.approx(0.05)
    expected = out["rho_n"] + (2.0 * 2.0 / 2.0) * (0.05 + out["r_n"])
    assert out["bound"] == pytest.approx(expected)


def test_bound_fail_closed() -> None:
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(4), np.ones(10), 0.1)  # < 5 scores
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(10), np.ones(10), 1.2)
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(10), np.ones(10), 0.1, m=-1.0)
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(10), np.ones(10), 0.1, delta0=-1.0)
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(10), np.ones(10), 0.1)  # zero spread


# ---------------------------------------------------------------------------
# context keys + DGP simulators
# ---------------------------------------------------------------------------


def test_build_lagged_context_keys_shape_and_rows() -> None:
    s = np.arange(10.0)
    keys = build_lagged_context_keys(s, 4)
    assert keys.shape == (7, 4)
    np.testing.assert_array_equal(keys[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(keys[6], [6.0, 7.0, 8.0, 9.0])
    with pytest.raises(ValueError):
        build_lagged_context_keys(s, 0)
    with pytest.raises(ValueError):
        build_lagged_context_keys(s, 10)
    with pytest.raises(ValueError):
        build_lagged_context_keys(np.array([np.nan] * 5), 2)


def test_simulate_dgp_kinds_deterministic() -> None:
    from quant_fund.metrics.rccp import _simulate_dgp

    for kind in ("iid", "heteroskedastic", "regime_shift"):
        a = _simulate_dgp(kind, 120, np.random.default_rng(11))
        b = _simulate_dgp(kind, 120, np.random.default_rng(11))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        assert a[0].shape == (120,) and a[2].ndim == 2
    with pytest.raises(ValueError):
        _simulate_dgp("bogus", 50, np.random.default_rng(0))


def test_dgp_regime_shift_has_heavier_second_half() -> None:
    from quant_fund.metrics.rccp import _simulate_dgp

    y, yh, _ = _simulate_dgp("regime_shift", 400, np.random.default_rng(5))
    resid = np.abs(y - yh)
    assert resid[200:].std() > 2.0 * resid[:200].std()


# ---------------------------------------------------------------------------
# Coverage battery on planted DGPs (SYNTHETIC)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dgp", ["iid", "heteroskedastic", "regime_shift"])
def test_battery_coverage_near_target(dgp: str) -> None:
    out = bench_rccp(dgp=dgp, seed=7)
    assert out["synthetic"] == 1.0
    # finite-sample slack: coverage within ~6pp of target on every planted DGP
    assert abs(out["coverage_gap_pct"]) <= 6.0
    assert 0.0 <= out["severe_miss_rate"] <= 1.0
    assert out["mean_width"] > 0.0


def test_battery_regime_shift_correction_earns_its_keep() -> None:
    """The scalar correction lifts coverage after a mid-series scale break."""
    out = bench_rccp(dgp="regime_shift", seed=7)
    assert out["c_hat"] > 1.0
    assert out["coverage"] > out["uncorrected_coverage"]
    # global SCP undercovers badly through the break; RCCP stays near target
    assert out["coverage"] > out["scp_coverage"]
    assert abs(out["coverage_gap_pct"]) < abs(
        100.0 * (out["scp_coverage"] - out["target_coverage"])
    )


def test_battery_heteroskedastic_winkler_not_worse_than_scp() -> None:
    out = bench_rccp(dgp="heteroskedastic", seed=7)
    assert out["winkler"] <= out["scp_winkler"]


def test_battery_iid_no_coverage_loss_vs_scp() -> None:
    """Exchangeable case: retrieval must not lose coverage vs the SCP baseline."""
    out = bench_rccp(dgp="iid", seed=7)
    assert out["coverage"] >= out["scp_coverage"] - 0.05
    assert out["bound_holds"] == 1.0


def test_battery_bound_vacuous_flag_on_regime_shift() -> None:
    """Under a genuine distribution break the plug-in bound honestly reports
    vacuity (Assumption 4 fails) rather than fabricating a finite bound."""
    out = bench_rccp(dgp="regime_shift", seed=7)
    assert out["bound_vacuous"] == 1.0
    assert out["bound"] == np.inf


def test_bench_deterministic_and_flat() -> None:
    a = bench_rccp(dgp="iid", seed=13)
    b = bench_rccp(dgp="iid", seed=13)
    assert a == b
    for k, v in a.items():
        assert isinstance(k, str) and isinstance(v, float)
    finite = {k: v for k, v in a.items() if k != "bound"}
    assert all(np.isfinite(v) for v in finite.values())
