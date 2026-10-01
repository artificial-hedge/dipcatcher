"""Tests for research/stocbench.py — SYNTHETIC correctness checks only.

StocBench (arXiv:2608.22309, Pfister-Holzschuh-Thuerey 2026) protocol legs on a
known Gaussian DGP: energy-distance closed forms (singleton Dirac ED = 2|a-b|,
identical ensembles = 0), budget-allocation rules (exact equal split, Neyman
sqrt-variance proportionality, min-2 floor, fail-closed under-budget), paired
score differentials (variance-normalized statistic, HAC t, block-bootstrap CI),
anytime-valid budget significance (excludes 0 iff the budget suffices), the
aleatoric/epistemic control-task split (non-translation rank correlation),
rollout invariant-measure drift, and the receipts-style experiment key. Seeded
Monte-Carlo only; proper scores, never P&L. Determinism pinned bit-for-bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.research.stocbench import (
    EvaluationTask,
    aleatoric_epistemic_split,
    allocate_budget,
    budget_significance,
    compare_samplers,
    control_task_scores,
    ensemble_energy_distance,
    experiment_key,
    rollout_invariant_drift,
    sampler_context_scores,
    sampler_distributional_errors,
    stocbench_benchmark,
)

_SEED = 20260823


def _dgp(seed: int = _SEED, n_t: int = 16, n_ref: int = 64):
    rng = np.random.default_rng(seed)
    t = np.arange(n_t, dtype=float)
    mu = 0.6 * np.sin(2.0 * np.pi * t / 8.0)
    sd = np.where(t < n_t / 2.0, 1.0, 1.5)
    eps = rng.standard_normal((n_t, 1))
    outcomes = mu[:, None] + sd[:, None] * eps
    refs = mu[:, None, None] + sd[:, None, None] * rng.standard_normal((n_t, n_ref, 1))
    return mu, sd, eps, outcomes, refs


def _task(seed: int = _SEED, n_t: int = 16, n_ref: int = 64) -> EvaluationTask:
    mu, sd, eps, outcomes, refs = _dgp(seed, n_t, n_ref)
    return EvaluationTask(
        references=refs, outcomes=outcomes, innovations=eps, name="SYNTHETIC_test"
    )


def _sampler(mu: np.ndarray, sd: np.ndarray, shift: float = 0.0, scale: float = 1.0):
    def _draw(rng: np.random.Generator, t: int, n: int) -> np.ndarray:
        return mu[t] + shift + sd[t] * scale * rng.standard_normal((n, 1))

    return _draw


def _control(mu: np.ndarray, sd: np.ndarray, shift: float = 0.0, scale: float = 1.0):
    def _map(eps: np.ndarray, t: int) -> np.ndarray:
        return np.array([mu[t] + shift + sd[t] * scale * float(eps[0])])

    return _map


def _draws(n_t: int, m: int = 8) -> np.ndarray:
    return np.full(n_t, m, dtype=np.int64)


# ---------------------------------------------------------------------------
# energy distance (Szekely-Rizzo empirical-measure V-statistic)
# ---------------------------------------------------------------------------


def test_energy_distance_identical_ensembles_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 1))
    assert ensemble_energy_distance(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_energy_distance_singleton_dirac_closed_form() -> None:
    # ED(delta_a, delta_b) = 2|a-b| (cross term 2|a-b|, self terms 0). n>=2 rows
    # required, so replicate the same atom: ED is still 2|a-b|.
    a = np.full((4, 1), 1.5)
    b = np.full((3, 1), -0.5)
    assert ensemble_energy_distance(a, b) == pytest.approx(4.0)


def test_energy_distance_nonnegative_and_symmetric_on_random() -> None:
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.standard_normal((30, 2))
        y = rng.standard_normal((25, 2)) + 0.2
        assert ensemble_energy_distance(x, y) >= -1e-12
        assert ensemble_energy_distance(x, y) == pytest.approx(
            ensemble_energy_distance(y, x), abs=1e-12
        )


def test_energy_distance_monotone_in_shift() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((60, 1))
    ed_small = ensemble_energy_distance(x, x + 0.3)
    ed_large = ensemble_energy_distance(x, x + 1.0)
    assert 0.0 < ed_small < ed_large


def test_energy_distance_equal_gaussian_laws_shrinks() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((400, 1))
    y = rng.standard_normal((400, 1))
    assert ensemble_energy_distance(x, y) < 0.05


def test_energy_distance_fail_closed() -> None:
    with pytest.raises(ValueError):
        ensemble_energy_distance(np.zeros((1, 1)), np.zeros((4, 1)))
    with pytest.raises(ValueError):
        ensemble_energy_distance(np.zeros((4, 1)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        ensemble_energy_distance(np.full((4, 1), np.nan), np.zeros((4, 1)))


# ---------------------------------------------------------------------------
# budget allocation (equal / Neyman)
# ---------------------------------------------------------------------------


def test_alloc_equal_exact_split() -> None:
    m = allocate_budget(200, 5, 2, rule="equal")
    assert (m == 20).all() and int(m.sum()) == 100


def test_alloc_equal_remainder_first_contexts() -> None:
    m = allocate_budget(2 * (5 * 10 + 3), 5, 2, rule="equal")
    assert int(m.sum()) == 53  # per-replicate spend: 106 // 2
    assert m.tolist() == [11, 11, 11, 10, 10]


def test_alloc_fail_closed_under_budget() -> None:
    with pytest.raises(ValueError):
        allocate_budget(2 * 5 * 2 - 1, 5, 2)
    with pytest.raises(ValueError):
        allocate_budget(10, 5, 2, rule="bogus")
    with pytest.raises(ValueError):
        allocate_budget(10.5, 5, 1)  # type: ignore[arg-type]


def test_alloc_neyman_proportional_to_sqrt_var() -> None:
    pilot = np.array([4.0, 1.0, 1.0, 1.0])
    m = allocate_budget(100, 4, 1, rule="neyman", pilot_variance=pilot)
    assert int(m.sum()) == 100
    assert m[0] > m[1]  # sqrt(4)=2 vs sqrt(1)=1
    # floor 2 each, then surplus 92 x sqrt shares (2/5,1/5,1/5,1/5)
    # = (36.8,18.4,18.4,18.4) -> floor+remainder -> (39,21,20,20)
    assert m.tolist() == [39, 21, 20, 20]


def test_alloc_neyman_min_floor_and_exact_spend() -> None:
    pilot = np.array([1e6, 1.0, 1.0, 1.0, 1.0])
    m = allocate_budget(5 * 20, 5, 1, rule="neyman", pilot_variance=pilot)
    assert int(m.sum()) == 100
    # floor 2 each + surplus 90 x w/W (w=[1000,1,1,1,1]) -> [92,2,2,2,2]
    assert m.tolist() == [92, 2, 2, 2, 2]


def test_alloc_neyman_requires_pilot() -> None:
    with pytest.raises(ValueError):
        allocate_budget(100, 4, 1, rule="neyman")
    with pytest.raises(ValueError):
        allocate_budget(100, 4, 1, rule="neyman", pilot_variance=np.zeros(4))
    with pytest.raises(ValueError):
        allocate_budget(100, 4, 1, rule="neyman", pilot_variance=np.ones(3))
    with pytest.raises(ValueError):
        allocate_budget(100, 4, 1, rule="neyman", pilot_variance=np.array([1.0, -1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# EvaluationTask validation
# ---------------------------------------------------------------------------


def test_task_fail_closed_shapes() -> None:
    mu, sd, eps, outcomes, refs = _dgp()
    with pytest.raises(ValueError):
        EvaluationTask(references=refs[:, :4], outcomes=outcomes, innovations=eps, name="x")
    with pytest.raises(ValueError):
        EvaluationTask(references=refs[:, :, 0], outcomes=outcomes, innovations=eps, name="x")
    with pytest.raises(ValueError):
        EvaluationTask(references=refs, outcomes=outcomes[:-1], innovations=eps, name="x")
    with pytest.raises(ValueError):
        EvaluationTask(references=refs, outcomes=outcomes, innovations=eps[:-1], name="x")
    with pytest.raises(ValueError):
        EvaluationTask(references=refs, outcomes=outcomes, name="")
    refs_bad = refs.copy()
    refs_bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        EvaluationTask(references=refs_bad, outcomes=outcomes, innovations=eps, name="x")


# ---------------------------------------------------------------------------
# sampler legs: proper score + distributional error + control
# ---------------------------------------------------------------------------


def test_context_scores_deterministic_and_ordering() -> None:
    mu, sd, eps, outcomes, refs = _dgp()
    task = _task()
    draws = _draws(task.n_contexts, 8)
    a = sampler_context_scores(task, _sampler(mu, sd), draws, n_replicates=3, seed=5)
    b = sampler_context_scores(task, _sampler(mu, sd), draws, n_replicates=3, seed=5)
    assert np.array_equal(a, b)
    shifted = sampler_context_scores(
        task, _sampler(mu, sd, shift=0.4), draws, n_replicates=3, seed=5
    )
    assert float(shifted.mean()) > float(a.mean())


def test_context_scores_fail_closed_on_bad_sampler() -> None:
    task = _task()
    draws = _draws(task.n_contexts, 8)

    def _bad_shape(rng: np.random.Generator, t: int, n: int) -> np.ndarray:
        return np.zeros((n + 1, 1))

    def _nan(rng: np.random.Generator, t: int, n: int) -> np.ndarray:
        return np.full((n, 1), np.nan)

    with pytest.raises(ValueError):
        sampler_context_scores(task, _bad_shape, draws, n_replicates=1, seed=0)
    with pytest.raises(ValueError):
        sampler_context_scores(task, _nan, draws, n_replicates=1, seed=0)
    with pytest.raises(ValueError):
        sampler_context_scores(task, _sampler(*_dgp()[:2]), draws[:-1], n_replicates=1, seed=0)
    with pytest.raises(ValueError):
        sampler_context_scores(task, _sampler(*_dgp()[:2]), _draws(task.n_contexts, 1), seed=0)


def test_distributional_errors_ordering_and_determinism() -> None:
    mu, sd, eps, outcomes, refs = _dgp()
    task = _task()
    draws = _draws(task.n_contexts, 16)
    oracle = sampler_distributional_errors(task, _sampler(mu, sd), draws, n_replicates=4, seed=9)
    shifted = sampler_distributional_errors(
        task, _sampler(mu, sd, shift=0.5), draws, n_replicates=4, seed=9
    )
    assert np.array_equal(
        oracle,
        sampler_distributional_errors(task, _sampler(mu, sd), draws, n_replicates=4, seed=9),
    )
    assert float(oracle.mean()) < float(shifted.mean())
    assert (oracle >= -1e-9).all()


def test_control_task_exact_reconstruction() -> None:
    mu, sd, eps, outcomes, refs = _dgp()
    task = _task()
    oracle_err = control_task_scores(task, _control(mu, sd))
    assert oracle_err == pytest.approx(np.zeros(task.n_contexts), abs=1e-12)
    shifted_err = control_task_scores(task, _control(mu, sd, shift=0.3))
    assert shifted_err == pytest.approx(np.full(task.n_contexts, 0.3))
    miss_err = control_task_scores(task, _control(mu, sd, scale=0.55))
    expected = np.abs(sd[:, None] * 0.45 * eps).reshape(-1)
    assert miss_err == pytest.approx(expected)


def test_control_task_requires_innovations() -> None:
    mu, sd, eps, outcomes, refs = _dgp()
    task = EvaluationTask(references=refs, outcomes=outcomes, name="no_innov")
    with pytest.raises(ValueError):
        control_task_scores(task, _control(mu, sd))

    def _bad_map(eps: np.ndarray, t: int) -> np.ndarray:
        return np.zeros(5)

    with pytest.raises(ValueError):
        control_task_scores(_task(), _bad_map)


# ---------------------------------------------------------------------------
# paired comparison: variance-normalized + HAC + block bootstrap
# ---------------------------------------------------------------------------


def test_compare_identical_streams() -> None:
    rng = np.random.default_rng(4)
    s = rng.normal(0.5, 0.1, (3, 20))
    cmp = compare_samplers(s, s.copy(), seed=0)
    assert cmp.mean_diff == 0.0 and cmp.var_normalized == 0.0
    assert not cmp.excludes_zero
    assert math.isnan(cmp.hac_t)


def test_compare_shifted_stream_excludes_zero() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(0.8, 0.15, (4, 24))
    b = rng.normal(0.5, 0.15, (4, 24))
    cmp = compare_samplers(a, b, seed=1)
    assert cmp.mean_diff > 0.2
    assert cmp.var_normalized > 1.0
    assert cmp.hac_t > 2.0
    assert cmp.boot_lo > 0.0 and cmp.excludes_zero
    assert cmp.as_dict()["n"] == float(96)


def test_compare_fail_closed() -> None:
    with pytest.raises(ValueError):
        compare_samplers(np.ones(7), np.ones(7))
    with pytest.raises(ValueError):
        compare_samplers(np.ones((2, 8)), np.ones((2, 9)))
    with pytest.raises(ValueError):
        compare_samplers(np.full(20, np.nan), np.ones(20))


def test_compare_deterministic() -> None:
    rng = np.random.default_rng(6)
    a, b = rng.normal(0.4, 0.1, 60), rng.normal(0.5, 0.1, 60)
    c1 = compare_samplers(a, b, n_boot=100, seed=7)
    c2 = compare_samplers(a, b, n_boot=100, seed=7)
    assert c1 == c2


# ---------------------------------------------------------------------------
# budget significance (anytime-valid CS over the diff stream)
# ---------------------------------------------------------------------------


def test_significance_strong_edge_excludes_zero() -> None:
    rng = np.random.default_rng(7)
    d = 0.5 + 0.1 * rng.standard_normal(60)
    sig = budget_significance(d, sigma=0.5, alpha=0.05)
    assert sig.significant_within_budget
    assert 0 < sig.first_excluding <= 60
    assert sig.final_lower > 0.0


def test_significance_null_stream_never_excludes() -> None:
    rng = np.random.default_rng(8)
    d = rng.standard_normal(80) * 0.2
    sig = budget_significance(d, sigma=0.5, alpha=0.05)
    assert not sig.significant_within_budget
    assert sig.first_excluding == -1
    assert sig.final_lower < 0.0 < sig.final_upper


def test_significance_fail_closed() -> None:
    with pytest.raises(ValueError):
        budget_significance(np.ones(1), sigma=1.0)
    with pytest.raises(ValueError):
        budget_significance(np.ones(10), sigma=0.0)
    with pytest.raises(ValueError):
        budget_significance(np.ones(10), sigma=1.0, alpha=1.5)
    with pytest.raises(ValueError):
        budget_significance(np.full(10, np.nan), sigma=1.0)


# ---------------------------------------------------------------------------
# aleatoric / epistemic split
# ---------------------------------------------------------------------------


def test_split_values_and_rank_corr() -> None:
    stoch = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]])
    ctrl = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    out = aleatoric_epistemic_split(stoch, ctrl)
    assert out["mean_stochastic"] == pytest.approx(stoch.mean())
    assert out["mean_control"] == pytest.approx(ctrl.mean())
    assert out["aleatoric_gap"] == pytest.approx(stoch.mean() - ctrl.mean())
    assert out["task_rank_corr"] == pytest.approx(1.0)
    out_rev = aleatoric_epistemic_split(stoch, ctrl[::-1].copy())
    assert out_rev["task_rank_corr"] == pytest.approx(-1.0)


def test_split_fail_closed() -> None:
    with pytest.raises(ValueError):
        aleatoric_epistemic_split(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        aleatoric_epistemic_split(np.ones(5), np.ones(6))


# ---------------------------------------------------------------------------
# rollout invariant-measure drift
# ---------------------------------------------------------------------------


def test_rollout_drift_zero_when_identical() -> None:
    rng = np.random.default_rng(10)
    ro = rng.standard_normal((4, 50, 1))
    drift = rollout_invariant_drift(ro, ro.copy())
    assert np.allclose(drift, 0.0)


def test_rollout_drift_grows_under_bias() -> None:
    rng = np.random.default_rng(11)
    h, n = 6, 200
    ref = rng.standard_normal((h, n, 1))
    good = np.stack([rng.standard_normal((n, 1)) for _ in range(h)])
    bad = np.stack([rng.standard_normal((n, 1)) + 0.1 * i for i in range(h)])
    d_good = rollout_invariant_drift(good, ref)
    d_bad = rollout_invariant_drift(bad, ref)
    assert d_bad[-1] > d_bad[0]
    assert d_bad[-1] > d_good[-1]


def test_rollout_drift_fail_closed() -> None:
    with pytest.raises(ValueError):
        rollout_invariant_drift(np.zeros((3, 10, 1)), np.zeros((4, 10, 1)))
    with pytest.raises(ValueError):
        rollout_invariant_drift(np.zeros((3, 10, 1)), np.zeros((3, 10, 2)))
    with pytest.raises(ValueError):
        rollout_invariant_drift(np.full((3, 10, 1), np.nan), np.zeros((3, 10, 1)))


# ---------------------------------------------------------------------------
# experiment key (receipts convention)
# ---------------------------------------------------------------------------


def test_experiment_key_deterministic_and_sensitive() -> None:
    p = {"b": 2, "a": [1, 2], "arr": np.array([1.0, 2.0])}
    k1 = experiment_key(p)
    k2 = experiment_key({"a": [1, 2], "arr": np.array([1.0, 2.0]), "b": 2})
    assert k1 == k2 and len(k1) == 64
    assert experiment_key({**p, "b": 3}) != k1


# ---------------------------------------------------------------------------
# stocbench_benchmark (seeded SYNTHETIC end-to-end)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench() -> dict[str, float]:
    return stocbench_benchmark()


def test_bench_keys_floats_and_clean_tokens(bench: dict[str, float]) -> None:
    assert bench
    for k, v in bench.items():
        assert k.startswith("stocbench_")
        assert isinstance(v, float) and np.isfinite(v)
        assert not any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in k.split("_"))


def test_bench_proper_score_ordering(bench: dict[str, float]) -> None:
    assert bench["stocbench_oracle_es_high"] < bench["stocbench_shifted_es_high"]
    assert bench["stocbench_oracle_es_high"] < bench["stocbench_misscaled_es_high"]
    assert bench["stocbench_derr_oracle"] < bench["stocbench_derr_shifted"]
    assert bench["stocbench_derr_oracle"] < 0.1


def test_bench_comparison_and_significance(bench: dict[str, float]) -> None:
    assert bench["stocbench_diff_shifted_mean"] > 0.0
    assert bench["stocbench_diff_shifted_vnorm"] > 0.0
    assert bench["stocbench_diff_shifted_hac_t"] > 2.0
    assert bench["stocbench_diff_shifted_boot_lo"] > 0.0
    assert bench["stocbench_sig_shifted_within_budget"] == 1.0
    assert bench["stocbench_sig_misscaled_within_budget"] == 1.0
    assert bench["stocbench_sig_shifted_first_excluding"] > 0.0
    assert bench["stocbench_sig_misscaled_first_excluding"] > 0.0


def test_bench_control_and_drift(bench: dict[str, float]) -> None:
    assert bench["stocbench_control_oracle"] == pytest.approx(0.0, abs=1e-9)
    assert bench["stocbench_control_shifted"] == pytest.approx(1.0, abs=1e-9)
    assert bench["stocbench_control_misscaled"] > bench["stocbench_control_oracle"]
    assert bench["stocbench_drift_shifted_last"] > bench["stocbench_drift_oracle_last"]
    assert -1.0 <= bench["stocbench_split_shifted_rank_corr"] <= 1.0


def test_bench_allocation(bench: dict[str, float]) -> None:
    assert bench["stocbench_alloc_equal_draws"] == 64.0
    assert bench["stocbench_alloc_neyman_min"] >= 2.0
    assert bench["stocbench_alloc_neyman_max"] >= bench["stocbench_alloc_neyman_min"]
    assert bench["stocbench_alloc_neyman_spent_per_rep"] == 48.0 * 64.0


def test_bench_deterministic() -> None:
    a = stocbench_benchmark(seed=99)
    b = stocbench_benchmark(seed=99)
    assert a == b
    assert a["stocbench_key_digest12"] == b["stocbench_key_digest12"]


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        stocbench_benchmark(draws_high=4, draws_low=8)
    with pytest.raises(ValueError):
        stocbench_benchmark(n_contexts=4)
    with pytest.raises(ValueError):
        stocbench_benchmark(alpha=0.0)
