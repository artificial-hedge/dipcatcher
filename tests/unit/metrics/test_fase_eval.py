"""FASE (arXiv:2609.32689) -- SYNTHETIC correctness tests.

Seeded simulations only: the eq. (1) normalised statistical distance against
hand-computed ρ-weighted values, eq. (2) retention margins on planted
partitions, two-pool promotion/eviction mechanics, the eq. (3) pairwise
ranking loss and its closed-form gradient, the paper's training schedule
(warmup / init epochs / periodic minibatches), GIFT-Eval §4.1 normalisation +
geometric-mean + average-tie-rank aggregation, the §5.2 cumulative-gain
curve, and the Appendix B online loop end to end. Correctness material,
never market evidence. No Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.fase_eval import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    N_FEATURES,
    BenchmarkInstance,
    EpisodicMemory,
    OnlineToolRanker,
    PolicyConfig,
    _pairwise_loss_and_grad,
    fase_benchmark,
    mase_score,
    mean_config_ranks,
    normalised_geometric_mean,
    normalised_statistical_distance,
    seasonal_naive_forecast,
    seasonal_naive_scale,
    self_evolution_curve,
    statistical_distance_row,
    statistical_features,
)

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _features(**kw: float) -> np.ndarray:
    """Build an 18-d representation with all features observed at 0.5."""
    v = np.full(N_FEATURES, 0.5)
    for name, val in kw.items():
        v[_IDX[name]] = val
    return v


# ---------------------------------------------------------------------------
# statistical_features (Appendix D, Table 7)
# ---------------------------------------------------------------------------


def test_features_shape_and_completeness() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, 128)
    cov = np.column_stack([x + rng.normal(0, 0.01, 128), rng.normal(0, 1, 128)])
    f = statistical_features(x, covariates=cov)
    assert f.shape == (N_FEATURES,)
    assert np.all(np.isfinite(f))


def test_features_table7_counts() -> None:
    assert N_FEATURES == 18
    assert len(FEATURE_GROUPS) == 8
    assert sum(len(names) for _, names in FEATURE_GROUPS) == N_FEATURES


def test_features_missingness_exact() -> None:
    x = np.arange(1, 21, dtype=float)
    x[5:8] = np.nan  # one run of 3 in 20 positions
    f = statistical_features(x)
    assert f[_IDX["missing_ratio"]] == pytest.approx(3 / 20)
    assert f[_IDX["longest_missing_block"]] == pytest.approx(3 / 20)
    assert f[_IDX["zero_ratio"]] == pytest.approx(0.0)


def test_features_zero_and_event_rate() -> None:
    x = np.concatenate([np.zeros(20), np.ones(20) * 2.0])
    f = statistical_features(x)
    assert f[_IDX["zero_ratio"]] == pytest.approx(0.5)
    # recent quarter = all nonzero vs prior with half zeros
    assert f[_IDX["recent_event_rate_change"]] > 0.0


def test_features_trend_and_level_shift() -> None:
    rng = np.random.default_rng(1)
    ramp = np.linspace(0, 10, 200) + rng.normal(0, 0.05, 200)
    f_trend = statistical_features(ramp)
    assert f_trend[_IDX["global_trend_strength"]] > 3.0
    flat = rng.normal(0.0, 1.0, 200)
    f_flat = statistical_features(flat)
    assert f_flat[_IDX["global_trend_strength"]] < 1.5
    shift = np.concatenate([rng.normal(0, 0.1, 160), rng.normal(5, 0.1, 40)])
    f_shift = statistical_features(shift)
    # shift/(global sigma incl. the jump) is bounded ~2; a flat series stays ~0
    assert f_shift[_IDX["recent_level_shift_strength"]] > 1.5
    assert abs(f_flat[_IDX["recent_level_shift_strength"]]) < 0.3


def test_features_direction_entropy_and_turning() -> None:
    alt = np.tile([0.0, 1.0], 50)  # diffs alternate +1,-1 -> all turns
    f = statistical_features(alt)
    assert f[_IDX["turning_behaviour_ratio"]] == pytest.approx(1.0)
    assert f[_IDX["direction_entropy"]] > 0.999  # 50 ups / 49 downs
    mono = np.linspace(0, 1, 100)  # all diffs + -> no turns, zero entropy
    f2 = statistical_features(mono)
    assert f2[_IDX["turning_behaviour_ratio"]] == pytest.approx(0.0)
    assert f2[_IDX["direction_entropy"]] == pytest.approx(0.0)


def test_features_periodicity() -> None:
    t = np.arange(200, dtype=float)
    sine = np.sin(2 * np.pi * t / 10.0)
    f_sine = statistical_features(sine)
    rng = np.random.default_rng(2)
    f_noise = statistical_features(rng.normal(0, 1, 200))
    assert f_sine[_IDX["recent_period_strength"]] > 0.8
    assert f_sine[_IDX["spectral_concentration"]] > f_noise[_IDX["spectral_concentration"]]


def test_features_covariates() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 120)
    cov = np.column_stack([base + rng.normal(0, 1e-6, 120)])
    f = statistical_features(base, covariates=cov)
    assert f[_IDX["strongest_covariate_relation"]] > 0.99
    assert np.isfinite(f[_IDX["covariate_relation_stability"]])
    # without covariates the group is unobserved (NaN), not fabricated
    f_none = statistical_features(base)
    assert np.isnan(f_none[_IDX["strongest_covariate_relation"]])
    assert np.isnan(f_none[_IDX["covariate_relation_stability"]])


def test_features_linear_fit_error_ratio() -> None:
    # Exact linear trend -> linear extrapolation perfect, ratio ~ 0
    f = statistical_features(np.linspace(0, 5, 60))
    assert f[_IDX["linear_fit_error_ratio"]] < 1e-3
    # White noise -> linear extrapolation worse than persistence, ratio > 1
    rng = np.random.default_rng(4)
    f2 = statistical_features(rng.normal(0, 1, 60))
    assert f2[_IDX["linear_fit_error_ratio"]] > 1.0


def test_features_constant_series_is_nan_not_crash() -> None:
    f = statistical_features(np.full(40, 3.0))
    assert f[_IDX["missing_ratio"]] == 0.0
    assert np.isnan(f[_IDX["global_trend_strength"]])


def test_features_fail_closed() -> None:
    with pytest.raises(ValueError):
        statistical_features(np.array([1.0, 2.0, 3.0]))  # too short
    with pytest.raises(ValueError):
        statistical_features(np.full(20, np.nan))  # nothing observed
    with pytest.raises(ValueError):
        statistical_features(np.ones(30), covariates=np.ones((29, 2)))  # misaligned


# ---------------------------------------------------------------------------
# normalised_statistical_distance (eq. 1)
# ---------------------------------------------------------------------------


def test_distance_self_is_zero_and_symmetric() -> None:
    s = _features()
    assert normalised_statistical_distance(s, s, np.ones(N_FEATURES)) == 0.0
    t = _features(missing_ratio=0.7, global_trend_strength=1.5)
    assert normalised_statistical_distance(s, t, np.ones(N_FEATURES)) == pytest.approx(
        normalised_statistical_distance(t, s, np.ones(N_FEATURES))
    )


def test_distance_exact_rho_value() -> None:
    # Only data_quality is shared: one feature equal, one off by sigma ->
    # group mean = (rho(0) + rho(1)) / 2 = 0.25; that is the only group.
    s = _features()
    t = _features(missing_ratio=1.5)  # |0.5 - 1.5| / sigma 1 = 1 -> rho = 0.5
    mask = np.zeros(N_FEATURES, dtype=bool)
    mask[[_IDX["missing_ratio"], _IDX["zero_ratio"]]] = True
    s[~mask] = np.nan
    t[~mask] = np.nan
    d = normalised_statistical_distance(s, t, np.ones(N_FEATURES))
    assert d == pytest.approx((0.0 + 0.5) / 2.0)


def test_distance_bounded_and_sigma_zero_rule() -> None:
    # sigma_q == 0: zero discrepancy -> 0 contribution, nonzero -> rho(inf) = 1
    s = _features()
    t = _features(missing_ratio=5.0)
    d = normalised_statistical_distance(s, t, np.zeros(N_FEATURES))
    # every feature differs by 4.5 except those at 0.5 == 0.5: group-mean mix
    assert 0.0 < d <= 1.0
    assert normalised_statistical_distance(s, s, np.zeros(N_FEATURES)) == 0.0


def test_distance_fail_closed_no_shared_group() -> None:
    s = np.full(N_FEATURES, np.nan)
    s[_IDX["missing_ratio"]] = 0.1  # only data_quality observed
    t = np.full(N_FEATURES, np.nan)
    t[_IDX["longest_missing_block"]] = 0.1  # only missingness observed
    with pytest.raises(ValueError, match="no shared"):
        normalised_statistical_distance(s, t, np.ones(N_FEATURES))


def test_distance_row_max_for_incomparable() -> None:
    q = _features()
    e = np.stack([q, np.full(N_FEATURES, np.nan)])
    d = statistical_distance_row(q, e, np.ones(N_FEATURES))
    assert d[0] == 0.0 and d[1] == 1.0
    assert statistical_distance_row(q, np.zeros((0, N_FEATURES)), np.ones(N_FEATURES)).size == 0


def test_distance_validation() -> None:
    with pytest.raises(ValueError):
        normalised_statistical_distance(_features(), _features(), -np.ones(N_FEATURES))
    with pytest.raises(ValueError):
        statistical_distance_row(_features(), np.zeros((2, 5)), np.ones(N_FEATURES))


# ---------------------------------------------------------------------------
# EpisodicMemory (Section 3.2)
# ---------------------------------------------------------------------------


def _mem_entry(mem: EpisodicMemory, feats: np.ndarray, tool: int = 0) -> None:
    mem.complete_instance(feats, tool, np.array([0.4, 0.2, 0.9]))


def test_memory_insert_retrieve_nearest() -> None:
    mem = EpisodicMemory(retrieve_k=1)
    # near differs in one feature of the 4-feature temporal group (diluted by
    # eq. 1's group mean); far differs in the 1-feature missingness group.
    near = _features(global_trend_strength=0.55)
    far = _features(longest_missing_block=9.0)
    _mem_entry(mem, far, tool=1)
    _mem_entry(mem, near, tool=2)
    out = mem.retrieve(_features())
    assert len(out) == 1 and out[0].invoked_tool == 2


def test_memory_recent_to_longterm_overflow() -> None:
    mem = EpisodicMemory(recent_capacity=3, longterm_capacity=2, retrieve_k=2)
    rng = np.random.default_rng(5)
    for _ in range(7):
        _mem_entry(mem, statistical_features(rng.normal(0, 1, 40)))
    assert mem.recent_size == 3
    assert mem.longterm_size == 2
    assert mem.n_active == 5  # 7 inserted, 2 dropped at the retention gate


def test_memory_retention_gate() -> None:
    # Direction 1: a zero-retention candidate cannot displace a high-retention
    # long-term entry. recent=1, lt=1, k=1 -> partition-nearest deltas only.
    mem = EpisodicMemory(recent_capacity=1, longterm_capacity=1, retrieve_k=1)
    fa = statistical_features(np.linspace(0, 1, 40))
    _mem_entry(mem, fa)  # id 0 -> recent
    # complete an identical query with A in context: A is partition-nearest and
    # singleton -> delta 1 - d = 1 -> retention 1; inserts id 1, evicts A to LT
    ctx = mem.retrieve(fa)
    mem.complete_instance(fa, 0, np.array([0.4, 0.2, 0.9]), context=ctx)
    e_a = next(e for e in mem._active() if e.entry_id == 0)
    assert e_a.retention == pytest.approx(1.0)
    # id 1 (retention 0) is evicted next -> gate rejects it, A survives
    _mem_entry(mem, _features(global_trend_strength=9.0))
    assert {e.entry_id for e in mem._active()} == {0, 2}

    # Direction 2: a candidate with higher retention replaces the weakest.
    mem2 = EpisodicMemory(recent_capacity=1, longterm_capacity=1, retrieve_k=10)
    mem2.complete_instance(fa, 0, np.array([0.4, 0.2, 0.9]))  # id 0 -> LT, ret 0
    fb = _features(global_trend_strength=9.0)
    mem2.complete_instance(fb, 0, np.array([0.4, 0.2, 0.9]))  # id 1 -> recent
    ctx = mem2.retrieve(fb)  # retrieves id 1
    mem2.complete_instance(fb, 0, np.array([0.4, 0.2, 0.9]), context=ctx)
    # id 1 retention 1.0; evicting it -> replaces weakest (id 0, retention 0)
    mem2.complete_instance(_features(missing_ratio=9.0), 0, np.array([0.4, 0.2, 0.9]))
    assert {e.entry_id for e in mem2._active()} == {1, 3}


def test_memory_retention_margin_exact() -> None:
    mem = EpisodicMemory(retrieve_k=10)
    q = _features()
    j1 = _features(missing_ratio=0.55)  # same partition (tool 0, same best set)
    j2 = _features(missing_ratio=0.8)  # farther
    # retention distances use the memory's own sigma over active entries {j1,j2}
    s = np.std(np.stack([j1, j2]), axis=0)
    d1 = normalised_statistical_distance(q, j1, s)
    d2 = normalised_statistical_distance(q, j2, s)
    mem.complete_instance(j1, 0, np.array([0.3, 0.5]))
    mem.complete_instance(j2, 0, np.array([0.3, 0.5]))  # same partition key
    ctx = mem.retrieve(q)  # both retrieved
    mem.complete_instance(q, 0, np.array([0.3, 0.5]), context=ctx)
    # j1 is partition-nearest -> delta = d(q, j2) - d(q, j1); running mean of 1
    e1 = mem._active()[0]
    e2 = mem._active()[1]
    assert e1.retention == pytest.approx(d2 - d1)
    assert e1.retention_count == 1
    assert e2.retention == 0.0  # not partition-nearest -> delta 0


def test_memory_best_tools_ties() -> None:
    mem = EpisodicMemory()
    e = mem.complete_instance(_features(), 0, np.array([0.1, 0.1, 0.5]))
    assert e.best_tools == frozenset({0, 1})
    assert e.point_ranks[0] == pytest.approx(1.5)


def test_memory_fail_closed() -> None:
    mem = EpisodicMemory()
    with pytest.raises(ValueError):
        mem.complete_instance(np.ones(5), 0, np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        mem.complete_instance(_features(), 0, np.array([-0.1, 0.2]))
    with pytest.raises(ValueError):
        mem.complete_instance(_features(), 9, np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        mem.complete_instance(_features(), 0, np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValueError):
        EpisodicMemory(recent_capacity=0)


def test_memory_determinism() -> None:
    def run() -> list[int]:
        mem = EpisodicMemory(recent_capacity=3, longterm_capacity=3, retrieve_k=2)
        rng = np.random.default_rng(11)
        for _ in range(9):
            f = statistical_features(rng.normal(0, 1, 60))
            ctx = mem.retrieve(f)
            mem.complete_instance(f, 0, np.array([0.4, 0.2]), context=ctx)
        return [e.entry_id for e in mem._active()]

    assert run() == run()


# ---------------------------------------------------------------------------
# OnlineToolRanker (Section 3.3 / Appendix C)
# ---------------------------------------------------------------------------


def _config(**kw: object) -> PolicyConfig:
    base = dict(
        n_tools=3,
        history_window=8,
        hidden_dims=(16, 8),
        warmup=10,
        init_epochs=2,
        update_every=5,
        update_batches=3,
        batch_size=8,
        seed=7,
    )
    base.update(kw)
    return PolicyConfig(**base)  # type: ignore[arg-type]


def test_observe_normalise_and_mask() -> None:
    r = OnlineToolRanker(_config())
    h = np.array([np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0])
    z = r.observe(h)
    assert z.shape == (16,)
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med, sd = np.median(obs), np.std(obs)
    expected = np.concatenate([(h - med) / np.where(sd > 0, sd, 1.0), np.isfinite(h).astype(float)])
    expected[:8] = np.where(np.isfinite(h), expected[:8], 0.0)
    np.testing.assert_allclose(z, expected)


def test_observe_padding_and_degenerate() -> None:
    r = OnlineToolRanker(_config(history_window=12))
    z = r.observe(np.ones(5))
    assert z.shape == (24,)
    assert np.all(z[12:] == np.concatenate([np.zeros(7), np.ones(5)]))
    assert np.all(z[:12] == 0.0)  # constant history -> zeroed normalised part
    with pytest.raises(ValueError):
        r.observe(np.full(10, np.nan))


def test_pairwise_loss_closed_form() -> None:
    # u = [0,0], ell = [1,2]: one pair, w = 1/1.5, margin 0 -> softplus ln2
    u = np.zeros((1, 2))
    ell = np.array([[1.0, 2.0]])
    loss, grad = _pairwise_loss_and_grad(u, ell)
    assert loss[0] == pytest.approx((1.0 / 1.5) * math.log(2.0))
    # coef = w * s * sigmoid(0) = (1/1.5) * 1 * 0.5 = 1/3; du0 +1/3, du1 -1/3
    np.testing.assert_allclose(grad, [[1.0 / 3.0, -1.0 / 3.0]])


def test_pairwise_loss_zero_when_errors_equal() -> None:
    u = np.array([[3.0, -1.0, 0.5]])
    ell = np.array([[0.7, 0.7, 0.7]])
    loss, grad = _pairwise_loss_and_grad(u, ell)
    assert loss[0] == pytest.approx(0.0)
    np.testing.assert_allclose(grad, np.zeros((1, 3)))


def test_pairwise_loss_monotone_in_wrong_score() -> None:
    ell = np.array([[0.1, 0.9]])  # tool 0 better -> wants u0 < u1
    good = _pairwise_loss_and_grad(np.array([[0.0, 1.0]]), ell)[0][0]
    bad = _pairwise_loss_and_grad(np.array([[1.0, 0.0]]), ell)[0][0]
    assert bad > good


def test_ranker_schedule_counts() -> None:
    cfg = _config(warmup=10, init_epochs=2, batch_size=8, update_every=5, update_batches=3)
    r = OnlineToolRanker(cfg)
    rng = np.random.default_rng(0)
    for _ in range(10):
        r.feedback(r.observe(rng.normal(0, 1, 20)), np.array([0.3, 0.1, 0.5]))
    # warmup hit at 10: 2 epochs x ceil(10/8)=2 batches -> 4 updates
    assert r.ready and r.n_updates == 4
    for _ in range(5):
        r.feedback(r.observe(rng.normal(0, 1, 20)), np.array([0.3, 0.1, 0.5]))
    # periodic at 15: +3 minibatches
    assert r.n_updates == 7


def test_ranker_scores_shape_and_learning() -> None:
    # Synthetic tool-error law: tool0 error = |mean(z_norm)|, tool1 = 1-|mean|;
    # the pairwise ranker should learn to prefer whichever fits each input.
    rng = np.random.default_rng(7)
    r = OnlineToolRanker(_config(history_window=8, warmup=40, init_epochs=6))
    for _ in range(60):
        x = rng.normal(0, 1, 8)
        z = r.observe(x)
        s = float(np.clip(abs(z[:8].mean()), 0, 1))
        ell = np.array([s, 1.0 - s, 1.5])  # tool 2 always worst
        r.feedback(z, ell)
    wins = 0
    trials = 200
    for _ in range(trials):
        x = rng.normal(0, 1, 8)
        z = r.observe(x)
        s = float(np.clip(abs(z[:8].mean()), 0, 1))
        scores = r.predict_scores(z)
        assert scores is not None
        true_best = int(np.argmin([s, 1.0 - s, 1.5]))
        wins += int(int(np.argmin(scores)) == true_best)
    assert wins / trials > 0.8


def test_ranker_fifo_capacity() -> None:
    r = OnlineToolRanker(_config(feedback_capacity=6, warmup=100))
    rng = np.random.default_rng(1)
    for _ in range(10):
        r.feedback(r.observe(rng.normal(0, 1, 12)), np.array([0.1, 0.2, 0.3]))
    assert r.pool_size == 6


def test_ranker_determinism() -> None:
    def run() -> np.ndarray:
        rng = np.random.default_rng(3)
        r = OnlineToolRanker(_config(warmup=12))
        for _ in range(16):
            r.feedback(r.observe(rng.normal(0, 1, 10)), rng.uniform(0.1, 1.0, 3))
        out = r.predict_scores(rng.normal(0, 1, 10))
        assert out is not None
        return out

    np.testing.assert_array_equal(run(), run())


def test_ranker_fail_closed() -> None:
    r = OnlineToolRanker(_config())
    with pytest.raises(ValueError):
        r.feedback(np.ones(16), np.array([0.1, -0.2, 0.3]))
    with pytest.raises(ValueError):
        r.feedback(np.ones(16), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        r.feedback(np.ones(10), np.array([0.1, 0.2, 0.3]))  # wrong z length
    with pytest.raises(ValueError):
        PolicyConfig(n_tools=1, history_window=4)
    with pytest.raises(ValueError):
        _config(hidden_dims=())


# ---------------------------------------------------------------------------
# GIFT-Eval protocol helpers (Section 4.1)
# ---------------------------------------------------------------------------


def test_seasonal_naive_forecast_repeats_cycle() -> None:
    f = seasonal_naive_forecast(np.arange(12, dtype=float), 5, season=4)
    np.testing.assert_array_equal(f, [8.0, 9.0, 10.0, 11.0, 8.0])
    f1 = seasonal_naive_forecast(np.array([2.0, 5.0]), 3, season=1)
    np.testing.assert_array_equal(f1, [5.0, 5.0, 5.0])


def test_seasonal_naive_scale_exact() -> None:
    # |2-1| + |4-2| + |8-4| = 7 over 3 pairs
    assert seasonal_naive_scale(np.array([1.0, 2.0, 4.0, 8.0]), 1) == pytest.approx(7.0 / 3.0)
    # NaN positions drop pairs honestly: pairs (t,t-1) both finite only once
    assert seasonal_naive_scale(np.array([1.0, np.nan, 4.0, 8.0]), 1) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        seasonal_naive_scale(np.array([1.0, 1.0, 1.0, 1.0]))  # zero scale
    with pytest.raises(ValueError):
        seasonal_naive_scale(np.array([1.0, 2.0]), 4)


def test_mase_score_exact() -> None:
    # |target - forecast| mean = (0.5 + 0.5) / 2 = 0.5; scale 2 -> MASE 0.25
    assert mase_score(np.array([1.0, 2.0]), np.array([1.5, 2.5]), 2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        mase_score(np.array([1.0]), np.array([1.0]), 0.0)


def test_normalised_geometric_mean() -> None:
    scores = {
        "base": np.array([2.0, 4.0]),
        "good": np.array([1.0, 1.0]),  # ratios 0.5, 0.25 -> geomean sqrt(1/8)
        "equal": np.array([2.0, 4.0]),
    }
    out = normalised_geometric_mean(scores, baseline="base")
    assert out["base"] == pytest.approx(1.0)
    assert out["equal"] == pytest.approx(1.0)
    assert out["good"] == pytest.approx(math.sqrt(0.125))
    with pytest.raises(ValueError):
        normalised_geometric_mean({"base": np.array([0.0]), "x": np.array([1.0])}, baseline="base")
    with pytest.raises(ValueError):
        normalised_geometric_mean({"x": np.array([1.0])}, baseline="missing")
    with pytest.raises(ValueError):
        normalised_geometric_mean({"base": np.array([1.0]), "x": np.array([-1.0])}, baseline="base")


def test_mean_config_ranks_ties() -> None:
    scores = {
        "a": np.array([1.0, 3.0]),
        "b": np.array([1.0, 1.0]),  # ties with a in cfg0 -> both rank 1.5
        "c": np.array([2.0, 2.0]),
    }
    ranks = mean_config_ranks(scores)
    assert ranks["a"] == pytest.approx((1.5 + 3.0) / 2)
    assert ranks["b"] == pytest.approx((1.5 + 1.0) / 2)
    assert ranks["c"] == pytest.approx((3.0 + 2.0) / 2)


def test_self_evolution_curve_exact() -> None:
    curve = self_evolution_curve(np.full(10, 0.5), np.full(10, 1.0))
    np.testing.assert_allclose(curve, np.full(10, 50.0))
    # ramp: agent halves error each step -> gain rises
    a = np.array([1.0, 0.5, 0.25, 0.125])
    c = self_evolution_curve(a, np.ones(4))
    assert np.all(np.diff(c) > 0.0) and c[-1] == pytest.approx((4.0 - a.sum()) / 4.0 * 100.0)
    with pytest.raises(ValueError):
        self_evolution_curve(np.ones(3), np.zeros(3))


# ---------------------------------------------------------------------------
# fase_benchmark (Appendix B online protocol)
# ---------------------------------------------------------------------------


def _make_instance(rng: np.random.Generator, h: int = 4) -> BenchmarkInstance:
    hist = rng.normal(0.0, 1.0, 60)
    target = rng.normal(0.0, 1.0, h)
    pf = np.stack(
        [
            target + rng.normal(0.0, 0.6, h),  # baseline (seasonal_naive stand-in)
            target + rng.normal(0.0, 1.5, h),
            target + rng.normal(0.0, 0.05, h),  # best tool
        ]
    )
    qf = np.stack(
        [
            np.stack([pf[j] + off for off in (-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2)])
            for j in range(3)
        ]
    )
    return BenchmarkInstance(history=hist, target=target, point_forecasts=pf, quantile_forecasts=qf)


def _bench_kwargs() -> dict[str, object]:
    return dict(
        tool_names=["seasonal_naive", "fm_a", "fm_b"],
        baseline_index=0,
        policy=PolicyConfig(
            n_tools=3,
            history_window=32,
            hidden_dims=(16, 8),
            warmup=8,
            init_epochs=2,
            update_every=4,
            update_batches=2,
            batch_size=8,
            seed=7,
        ),
    )


def test_fase_benchmark_baseline_is_one_and_agent_beats() -> None:
    rng = np.random.default_rng(42)
    insts = [_make_instance(rng) for _ in range(40)]
    res = fase_benchmark(insts, **_bench_kwargs())  # type: ignore[arg-type]
    assert res["seasonal_naive_norm_mase"] == pytest.approx(1.0)
    assert res["fm_b_norm_mase"] < 0.2
    assert res["agent_norm_mase"] < res["seasonal_naive_norm_mase"]
    assert res["agent_norm_mae"] < 1.0
    assert res["self_evolution_gain_final"] > 0.0
    assert res["n_instances"] == 40.0 and res["n_configs"] == 1.0
    assert res["policy_ready"] == 1.0 and res["n_policy_updates"] > 0.0
    assert res["fm_b_rank_mase"] == pytest.approx(1.0)


def test_fase_benchmark_base_variant_is_baseline() -> None:
    rng = np.random.default_rng(5)
    insts = [_make_instance(rng) for _ in range(12)]
    res = fase_benchmark(insts, variant="base", **_bench_kwargs())  # type: ignore[arg-type]
    assert res["agent_norm_mase"] == pytest.approx(1.0)
    assert res["self_evolution_gain_final"] == pytest.approx(0.0)
    assert res["n_policy_updates"] == 0.0


def test_fase_benchmark_ablations_and_no_quantiles() -> None:
    rng = np.random.default_rng(6)
    insts = [_make_instance(rng) for _ in range(15)]
    for inst in insts:
        object.__setattr__(inst, "quantile_forecasts", None)
    res = fase_benchmark(insts, variant="memory_only", **_bench_kwargs())  # type: ignore[arg-type]
    assert "agent_norm_crps" not in res  # no probabilistic feedback -> no CRPS
    assert res["memory_recent_size"] == 15.0
    res2 = fase_benchmark(
        [_make_instance(rng) for _ in range(15)], variant="policy_only", **_bench_kwargs()
    )  # type: ignore[arg-type]
    assert res2["memory_recent_size"] == 0.0


def test_fase_benchmark_determinism() -> None:
    def run() -> dict[str, float]:
        rng = np.random.default_rng(9)
        insts = [_make_instance(rng) for _ in range(25)]
        return fase_benchmark(insts, **_bench_kwargs())  # type: ignore[arg-type]

    assert run() == run()


def test_fase_benchmark_multi_config() -> None:
    rng = np.random.default_rng(8)
    cfgs = [[_make_instance(rng) for _ in range(12)] for _ in range(2)]
    res = fase_benchmark(cfgs, **_bench_kwargs())  # type: ignore[arg-type]
    assert res["n_configs"] == 2.0 and res["n_instances"] == 24.0
    # state reinitialised per config: two recent pools of 12 each -> sum 24
    assert res["memory_recent_size"] == 24.0


def test_fase_benchmark_fail_closed() -> None:
    rng = np.random.default_rng(0)
    insts = [_make_instance(rng) for _ in range(5)]
    with pytest.raises(ValueError):
        fase_benchmark([], **_bench_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fase_benchmark(insts, variant="nope", **_bench_kwargs())  # type: ignore[arg-type]
    bad = BenchmarkInstance(
        history=insts[0].history, target=np.ones(3), point_forecasts=insts[0].point_forecasts
    )
    with pytest.raises(ValueError):
        fase_benchmark([bad], **_bench_kwargs())  # type: ignore[arg-type]
    kw = _bench_kwargs()
    kw["baseline_index"] = 9
    with pytest.raises(ValueError):
        fase_benchmark(insts, **kw)  # type: ignore[arg-type]
