"""Tests for models/hierarchical_conformal.py — GHCP (arXiv:2608.15500).

Reductions are pinned as closed-form equalities; coverage results are
SYNTHETIC correctness evidence, never market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.hierarchical_conformal import (
    MAX_GROUPS,
    GHCPResult,
    bench_ghcp,
    donor_pool,
    hcp_predict,
    weighted_measure_quantile,
)


def _hier_gaussian_groups(
    seed: int, sizes: list[int], gamma: float = 5.0, sigma: float = 1.0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Group effects + within-group noise; returns reference groups and test stream."""
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, gamma, size=len(sizes) + 1)
    groups = [
        np.asarray(rng.normal(effects[j], sigma, size=n), dtype=np.float64)
        for j, n in enumerate(sizes)
    ]
    return groups, effects


def _assert_results_equal(a: GHCPResult, b: GHCPResult) -> None:
    scalars = (
        a.lower == b.lower
        and a.upper == b.upper
        and a.threshold == b.threshold
        and a.center == b.center
        and a.trivial == b.trivial
        and a.donor_index == b.donor_index
        and a.s_pool == b.s_pool
        and a.s_cal == b.s_cal
        and a.s_train == b.s_train
        and a.surrogate_size == b.surrogate_size
        and a.n_initial == b.n_initial
        and a.tau == b.tau
        and a.lam == b.lam
        and a.alpha == b.alpha
        and a.eta == b.eta
        and a.seed == b.seed
        and a.test_weight == b.test_weight
        and a.inf_atoms == b.inf_atoms
        and a.inf_weight == b.inf_weight
    )
    assert scalars
    assert np.array_equal(a.cal_scores, b.cal_scores)
    assert np.array_equal(a.cal_weights, b.cal_weights)
    assert np.array_equal(a.test_scores, b.test_scores)


def test_reduction_zero_initial_matches_hcp_closed_form() -> None:
    """o = 0: nu_donor IS the Lee et al. (2026) HCP measure on S_cal."""
    sizes = [16] * 16
    y_groups, _ = _hier_gaussian_groups(3, sizes)
    alpha = 0.20
    res = hcp_predict(y_groups, np.empty(0), alpha=alpha, eta=0.5, seed=5)

    # Structure: |S_eta| = ceil(0.5 * 16) = 8; one donor; seven calibrators.
    assert len(res.s_pool) == 8
    assert res.donor_index in res.s_pool
    assert res.s_cal == tuple(j for j in res.s_pool if j != res.donor_index)
    assert len(res.s_cal) == 7
    assert len(res.s_train) == 8
    assert set(res.s_train).isdisjoint(res.s_pool)
    assert res.tau == 0
    assert res.lam == 0.0
    assert res.surrogate_size == 16
    assert res.n_initial == 0

    # HCP group weights 1/(K1+1) and the +inf placeholder are exact here:
    # sizes 16 and S_size 8 are powers of two, so every atom is 1/128.
    assert res.inf_weight == 0.125
    assert res.inf_atoms == 16
    assert np.all(res.cal_weights == 1.0 / 128.0)
    assert res.cal_scores.size == 7 * 16

    # Independent closed-form HCP quantile on the realized split.
    mu_glob = float(np.mean(np.concatenate([y_groups[j] for j in res.s_train])))
    scores = np.concatenate([np.abs(y_groups[j] - mu_glob) for j in res.s_cal])
    assert np.array_equal(res.cal_scores, scores)
    ordered = np.sort(scores)
    k1 = len(res.s_cal)
    rank_direct = int(np.ceil((1.0 - alpha) * 128.0))
    rank_hcp_level = int(np.ceil(scores.size * (1.0 - alpha) * (k1 + 1) / k1))
    assert rank_direct == rank_hcp_level == 103
    assert res.threshold == ordered[rank_direct - 1]
    assert not res.trivial
    assert (
        weighted_measure_quantile(res.cal_scores, res.cal_weights, res.inf_weight, alpha)
        == res.threshold
    )
    assert res.lower == res.center - res.threshold
    assert res.upper == res.center + res.threshold


def test_reduction_zero_initial_trivial_when_alpha_below_placeholder() -> None:
    """o = 0 with alpha < 1/(K1+1): the HCP placeholder mass forces +inf."""
    sizes = [16] * 16
    y_groups, _ = _hier_gaussian_groups(3, sizes)
    res = hcp_predict(y_groups, np.empty(0), alpha=0.05, eta=0.5, seed=5)
    assert res.inf_weight == 0.125 > 0.05
    assert res.threshold == float("inf")
    assert res.trivial
    assert res.lower == float("-inf")
    assert res.upper == float("inf")


def test_reduction_no_donors_matches_within_group_split_conformal() -> None:
    """S = {} (every group <= o): exactly within-test-group split conformal."""
    sizes = [12, 17, 20, 25, 30]
    y_groups, effects = _hier_gaussian_groups(4, sizes)
    rng = np.random.default_rng(55)
    o = 30
    y_test = np.asarray(rng.normal(effects[-1], 1.0, size=o), dtype=np.float64)
    alpha = 0.10
    res = hcp_predict(y_groups, y_test, alpha=alpha, eta=0.5, seed=9)

    assert res.s_pool == ()
    assert res.s_cal == ()
    assert res.donor_index is None
    assert res.s_train == tuple(range(len(sizes)))
    assert res.surrogate_size == o + 1
    tau = o // 2
    assert res.tau == tau == 15
    assert res.lam == 0.75
    assert res.test_weight == 1.0 / 16.0
    assert res.inf_weight == 1.0 / 16.0
    assert res.inf_atoms == 1

    # Independent closed-form: split conformal on the held-out half.
    mu_glob = float(np.mean(np.concatenate(y_groups)))
    mu_loc = float(np.mean(y_test[:tau]))
    mu_tilde = (1.0 - res.lam) * mu_glob + res.lam * mu_loc
    holdout = np.abs(y_test[tau:o] - mu_tilde)
    assert np.array_equal(res.test_scores, holdout)
    n1 = holdout.size
    rank = int(np.ceil((n1 + 1) * (1.0 - alpha)))
    assert res.threshold == np.sort(holdout)[rank - 1]
    assert res.threshold == conformal_quantile(holdout, alpha)
    assert res.center == mu_tilde
    assert not res.trivial


def test_reduction_no_donors_trivial_when_level_unattainable() -> None:
    """Remark 2.2 edge: o = 4 gives 3 atoms; alpha < 1/3 must return +inf."""
    sizes = [2, 3, 4]
    y_groups, _ = _hier_gaussian_groups(6, sizes)
    y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = hcp_predict(y_groups, y_test[:4], alpha=0.10, eta=0.5, seed=1)
    assert res.s_pool == ()
    assert res.inf_atoms == 1
    assert res.inf_weight == pytest.approx(1.0 / 3.0)
    assert res.threshold == float("inf")
    assert res.trivial


def test_reduction_k0_matches_within_group_split_conformal() -> None:
    """K = 0: pure within-group split conformal, local mean only (lam = 1)."""
    rng = np.random.default_rng(6)
    y_test = np.asarray(rng.normal(size=8), dtype=np.float64)
    res = hcp_predict([], y_test, alpha=0.20, seed=0)
    assert res.s_pool == ()
    assert res.s_cal == ()
    assert res.s_train == ()
    assert res.donor_index is None
    assert res.surrogate_size == 9
    assert res.tau == 4
    assert res.lam == 1.0
    assert res.test_weight == 0.2
    assert res.inf_weight == 0.2

    loc = float(np.mean(y_test[:4]))
    holdout = np.abs(y_test[4:8] - loc)
    assert np.array_equal(res.test_scores, holdout)
    assert res.center == loc
    assert res.threshold == conformal_quantile(holdout, 0.20)
    assert not res.trivial

    # o = 1, K = 0: a single calibration atom cannot reach 1 - alpha = 0.9.
    tiny = hcp_predict([], y_test[:1], alpha=0.10, seed=0)
    assert tiny.threshold == float("inf")
    assert tiny.trivial


def test_single_reference_group_no_initial_is_trivial() -> None:
    """Degenerate single group with o = 0: nu = delta_{+inf}, honest trivial set."""
    y_groups = [np.arange(1.0, 7.0)]
    res = hcp_predict(y_groups, np.empty(0), alpha=0.10, seed=2)
    assert res.s_pool == (0,)
    assert res.donor_index == 0
    assert res.s_cal == ()
    assert res.s_train == ()
    assert res.surrogate_size == 6
    assert res.cal_scores.size == 0
    assert res.test_scores.size == 0
    assert res.inf_atoms == 6
    assert res.inf_weight == pytest.approx(1.0)
    assert res.threshold == float("inf")
    assert res.trivial
    assert res.lower == float("-inf")
    assert res.upper == float("inf")


def test_donor_pool_selection_rules() -> None:
    sizes = np.array([16, 5, 9, 30, 12, 7, 25, 3], dtype=np.int64)
    # eta = 0 keeps every compatible group (unrestricted pool, Sec 2.1).
    full = donor_pool(sizes, 5, 0.0, np.random.default_rng(0))
    assert set(full.tolist()) == {0, 2, 3, 4, 5, 6}
    # eta = 0.5 keeps the ceil(0.5 * 8) = 4 smallest compatible sizes.
    half = donor_pool(sizes, 5, 0.5, np.random.default_rng(0))
    assert set(half.tolist()) == {0, 2, 4, 5}
    # No compatible donors -> empty pool.
    empty = donor_pool(sizes, 100, 0.5, np.random.default_rng(0))
    assert empty.size == 0
    # Cutoff ties broken by the seeded rng, deterministically per seed.
    tied = np.full(10, 21, dtype=np.int64)
    a = donor_pool(tied, 5, 0.5, np.random.default_rng(1))
    b = donor_pool(tied, 5, 0.5, np.random.default_rng(1))
    c = donor_pool(tied, 5, 0.5, np.random.default_rng(2))
    assert a.size == b.size == c.size == 5
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    with pytest.raises(ValueError):
        donor_pool(sizes, 5, 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        donor_pool(sizes, 5, -0.1, np.random.default_rng(0))


def test_weighted_measure_quantile_matches_conformal_order_statistic() -> None:
    rng = np.random.default_rng(8)
    scores = np.asarray(rng.normal(size=40), dtype=np.float64)
    alpha = 0.10
    # Uniform weights 1/(n+1) plus a 1/(n+1) atom at +inf == split conformal.
    q = weighted_measure_quantile(scores, np.full(40, 1.0 / 41.0), 1.0 / 41.0, alpha)
    assert q == conformal_quantile(scores, alpha)


def test_weighted_measure_quantile_boundary_and_trivial() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.full(3, 1.0 / 6.0)
    # inf weight == alpha: the level is attained exactly at the largest atom.
    assert weighted_measure_quantile(scores, weights, 0.5, 0.5) == 3.0
    # inf weight > alpha: unattainable -> +inf (no clipping).
    assert weighted_measure_quantile(scores, weights, 0.6, 0.5) == float("inf")
    # No finite atoms at all -> +inf.
    empty = np.empty(0, dtype=float)
    assert weighted_measure_quantile(empty, empty, 1.0, 0.5) == float("inf")
    with pytest.raises(ValueError):
        weighted_measure_quantile(scores, weights, 0.1, 0.0)
    with pytest.raises(ValueError):
        weighted_measure_quantile(scores, weights, 0.1, 1.0)
    with pytest.raises(ValueError):
        weighted_measure_quantile(scores, weights[:2], 0.1, 0.2)
    with pytest.raises(ValueError):
        weighted_measure_quantile(scores, np.full(3, -0.1), 0.1, 0.2)
    with pytest.raises(ValueError):
        weighted_measure_quantile(scores, weights, -0.1, 0.2)
    with pytest.raises(ValueError):
        weighted_measure_quantile(np.array([1.0, np.inf]), np.full(2, 0.25), 0.25, 0.25)


def test_determinism_same_seed_and_global_rng_independence() -> None:
    sizes = [10 + 2 * j for j in range(12)]
    y_groups, effects = _hier_gaussian_groups(10, sizes)
    rng = np.random.default_rng(11)
    y_test = np.asarray(rng.normal(effects[-1], 1.0, size=9), dtype=np.float64)

    a = hcp_predict(y_groups, y_test[:6], alpha=0.15, eta=0.5, seed=17)
    b = hcp_predict(y_groups, y_test[:6], alpha=0.15, eta=0.5, seed=17)
    _assert_results_equal(a, b)

    np.random.seed(999)
    _ = np.random.random(16)
    c = hcp_predict(y_groups, y_test[:6], alpha=0.15, eta=0.5, seed=17)
    _assert_results_equal(a, c)

    assert not a.trivial
    d = hcp_predict(y_groups, y_test[:6], alpha=0.15, eta=0.5, seed=18)
    assert d.donor_index != a.donor_index
    assert d.s_cal != a.s_cal
    assert d.inf_weight != a.inf_weight


def test_covariate_ridge_global_predictor_path() -> None:
    rng = np.random.default_rng(21)
    k, n, d, o = 6, 14, 2, 6
    beta = np.array([2.0, -1.0])
    x_groups = [np.asarray(rng.normal(size=(n, d)), dtype=np.float64) for _ in range(k)]
    effects = rng.normal(0.0, 1.0, size=k + 1)
    y_groups = [
        np.asarray(x @ beta + effects[j] + rng.normal(0.0, 0.5, size=n), dtype=np.float64)
        for j, x in enumerate(x_groups)
    ]
    x_test = np.asarray(rng.normal(size=(o + 1, d)), dtype=np.float64)
    y_test = np.asarray(
        x_test @ beta + effects[-1] + rng.normal(0.0, 0.5, size=o + 1), dtype=np.float64
    )
    kwargs = {
        "x_groups": x_groups,
        "x_test_initial": x_test[:o],
        "x_test_target": x_test[o],
        "alpha": 0.30,
        "eta": 0.5,
        "ridge": 1e-3,
        "seed": 4,
    }
    res = hcp_predict(y_groups, y_test[:o], **kwargs)
    assert not res.trivial
    assert np.isfinite(res.threshold)
    assert res.lower <= res.center <= res.upper
    assert res.upper - res.lower == pytest.approx(2.0 * res.threshold)
    again = hcp_predict(y_groups, y_test[:o], **kwargs)
    _assert_results_equal(res, again)


def test_fail_closed_validation() -> None:
    rng = np.random.default_rng(0)
    y_groups = [np.asarray(rng.normal(size=6), dtype=np.float64) for _ in range(3)]
    y_test = np.asarray(rng.normal(size=4), dtype=np.float64)

    for bad_alpha in (0.0, 1.0, -0.1, np.nan):
        with pytest.raises(ValueError):
            hcp_predict(y_groups, y_test, alpha=bad_alpha)
    for bad_eta in (1.0, -0.1, np.nan):
        with pytest.raises(ValueError):
            hcp_predict(y_groups, y_test, eta=bad_eta)
    with pytest.raises(ValueError):
        hcp_predict(y_groups, y_test, ridge=-1e-6)
    with pytest.raises(ValueError):
        hcp_predict(y_groups, y_test, ridge=np.nan)
    with pytest.raises(ValueError):
        hcp_predict([], np.empty(0))
    with pytest.raises(ValueError):
        hcp_predict([np.empty(0)] + y_groups, y_test)
    with pytest.raises(ValueError):
        hcp_predict([np.array([1.0, np.nan])] + y_groups, y_test)
    with pytest.raises(ValueError):
        hcp_predict(y_groups, np.array([1.0, np.inf, 3.0, 4.0]))
    with pytest.raises(ValueError):
        hcp_predict([np.asarray(rng.normal(size=(2, 2)), dtype=np.float64)], y_test)
    with pytest.raises(ValueError):
        hcp_predict([np.zeros(1)] * (MAX_GROUPS + 1), np.empty(0))
    x_bad = [np.zeros((6, 2)) for _ in range(3)]
    with pytest.raises(ValueError):
        hcp_predict(y_groups, y_test, x_groups=x_bad)
    with pytest.raises(ValueError):
        hcp_predict(
            y_groups,
            y_test,
            x_groups=x_bad,
            x_test_initial=np.zeros((4, 2)),
            x_test_target=np.zeros(3),
        )
    with pytest.raises(ValueError):
        hcp_predict(y_groups, y_test, x_test_target=np.zeros(2))


def test_synthetic_coverage_and_width_shrink_across_m() -> None:
    row = bench_ghcp(n_reps=300, seed=11)
    assert row["synthetic_dgp"] == "synthetic_hierarchical_gaussian"
    assert row["synthetic_claim"] == "research_metric_only"
    nominal = 1.0 - float(row["synthetic_alpha"])
    tol = 0.04
    widths = []
    for m in (0, 1, 3, 10):
        coverage = float(row[f"synthetic_coverage_m{m}"])
        assert coverage >= nominal - tol, f"m={m}: coverage {coverage}"
        assert float(row[f"synthetic_trivial_share_m{m}"]) == 0.0
        widths.append(float(row[f"synthetic_mean_width_m{m}"]))
    assert all(np.isfinite(widths))
    # The value of the generalization: sets shrink as the in-group sample grows.
    assert widths[3] < 0.75 * widths[0]
    for earlier, later in zip(widths, widths[1:], strict=False):
        assert later <= earlier * 1.05
    assert float(row["synthetic_width_shrinks"]) == 1.0
    assert float(row["synthetic_min_coverage"]) >= nominal - tol


def test_synthetic_coverage_second_seed_and_level() -> None:
    row = bench_ghcp(n_reps=200, seed=77, alpha=0.20)
    nominal = 0.80
    for m in (0, 1, 3, 10):
        assert float(row[f"synthetic_coverage_m{m}"]) >= nominal - 0.04
        assert float(row[f"synthetic_trivial_share_m{m}"]) == 0.0
    assert float(row["synthetic_width_shrinks"]) == 1.0


def test_bench_determinism_and_key_hygiene() -> None:
    a = bench_ghcp(n_reps=20, seed=3)
    b = bench_ghcp(n_reps=20, seed=3)
    assert a == b
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav")
    for key in a:
        lowered = key.lower()
        assert all(token not in lowered for token in forbidden)
    assert a["synthetic_seed"] == 3.0
    assert a["synthetic_n_reps"] == 20.0
    assert 0.0 <= float(a["synthetic_min_coverage"]) <= 1.0
    with pytest.raises(ValueError):
        bench_ghcp(n_reps=0)
    with pytest.raises(ValueError):
        bench_ghcp(m_values=(-1, 2))
    with pytest.raises(ValueError):
        bench_ghcp(base_size=2, m_values=(0, 10))
