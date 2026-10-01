"""Tests for adaptive_eps: e-PS (Lin, Ma, Ren & Wei 2026, arXiv:2609.26651).

All worlds are seeded SYNTHETIC planted Gaussian worlds (correctness tests,
never market evidence). Every random draw flows through
np.random.default_rng / SeedSequence, so runs are deterministic given seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.adaptive_eps import (
    POLICIES,
    EPsSelector,
    GaussianWorld,
    StepSnapshot,
    bench_eps_efficiency,
    eps_run,
    fixed_design_ebh,
    make_composite_vs_simple_world,
    make_oracle,
    make_simple_vs_composite_world,
    make_simple_vs_simple_world,
)

BoolArray = NDArray[np.bool_]

SEED = 20260930
ALPHA = 0.1


# ------------------------------------------------------------ fixtures ----


def _svs_world(n_hyp: int = 12, k_nonnull: int = 3, seed: int = SEED) -> GaussianWorld:
    return make_simple_vs_simple_world(
        n_hyp, k_nonnull, effect=0.8, stagger=0.2, var=1.0, seed=seed
    )


def _cvs_world(n_hyp: int = 12, k_nonnull: int = 3, seed: int = SEED + 1) -> GaussianWorld:
    return make_composite_vs_simple_world(
        n_hyp, k_nonnull, effect=0.8, stagger=0.2, null_spread=0.3, var=1.0, seed=seed
    )


def _svc_world(n_hyp: int = 12, k_nonnull: int = 3, seed: int = SEED + 2) -> GaussianWorld:
    return make_simple_vs_composite_world(
        n_hyp, k_nonnull, effect=0.9, stagger=0.25, var=1.0, seed=seed
    )


_WORLD_FACTORIES = (_svs_world, _cvs_world, _svc_world)


def _discovery_rule(nonnull: BoolArray):
    def rule(snap: StepSnapshot) -> bool:
        return bool(np.all(snap.rejected[nonnull]))

    return rule


# ------------------------------------------------------- EPsSelector ----


def test_selector_initialization_phase_is_round_robin() -> None:
    sel = EPsSelector(5, seed=SEED)
    for t in range(1, 6):
        assert sel.t == t
        arm = sel.select()
        assert arm == t - 1  # Algorithm 1 line 4: A_t = t (0-based here)
        sel.observe(arm, 0.05)
    assert sel.t == 6
    assert sel.counts.tolist() == [1, 1, 1, 1, 1]


def test_selector_posterior_sampling_excludes_rejected() -> None:
    # Proposition 3.1 support condition: supp(pi_t) subset [K] \ R_{t-1}.
    sel = EPsSelector(6, seed=SEED)
    for arm in range(6):
        sel.observe(arm, 0.1 * arm)
    sel.mark_rejected(np.asarray([False, False, True, False, False, False]))
    seen = {sel.select() for _ in range(200)}
    assert 2 not in seen
    assert seen <= {0, 1, 3, 4, 5}


def test_selector_posterior_prefers_high_log_e_arms() -> None:
    # Exploitation: with tiny posterior variance, the arm with the largest
    # mean log increment is selected (Thompson-style argmax degenerates).
    sel = EPsSelector(4, seed=SEED, initial_variance=1e-12)
    increments = [(-1.0, 0.5, 0.0, -0.2)]
    for arm in range(4):
        sel.observe(arm, increments[0][arm])
        sel.observe(arm, increments[0][arm])
    picks = [sel.select() for _ in range(50)]
    assert all(p == 1 for p in picks)


def test_selector_mean_and_variance_proxy() -> None:
    sel = EPsSelector(3, seed=SEED, variance_inflation=1.1, initial_variance=2.0)
    z = np.asarray([0.2, -0.4, 0.6, 0.0])
    for v in z:
        sel.observe(0, float(v))
    sel.observe(1, 0.5)  # single increment -> fallback proxy
    # m-hat = log(E)/(n v 1): empirical mean of log increments.
    assert sel.means()[0] == pytest.approx(float(z.mean()))
    assert sel.means()[1] == pytest.approx(0.5)
    assert sel.means()[2] == 0.0  # unsampled: log E = 0, n v 1 = 1
    # v = 1.1 * sample variance of the arm's log increments (paper Section 6).
    assert sel.variance_proxies()[0] == pytest.approx(1.1 * float(np.var(z, ddof=1)))
    assert sel.variance_proxies()[1] == pytest.approx(2.0)  # n < 2 fallback
    assert sel.posterior_sds()[0] == pytest.approx(np.sqrt(1.1 * float(np.var(z, ddof=1)) / 4.0))


def test_selector_variance_fn_override() -> None:
    # Theorem-style deterministic proxy v(k, n) (e.g. rho^2 v kappa^2, Eq. 8).
    sel = EPsSelector(3, seed=SEED, variance_fn=lambda k, n: 0.5 + 0.1 * k + 0.0 * n)
    sel.observe(0, 0.3)
    sel.observe(0, -0.1)
    v = sel.variance_proxies()
    assert v[0] == pytest.approx(0.5)
    assert v[1] == pytest.approx(0.6)
    assert v[2] == pytest.approx(0.7)


def test_selector_nesting_violation_fails_closed() -> None:
    sel = EPsSelector(4, seed=SEED)
    sel.mark_rejected(np.asarray([True, False, True, False]))
    with pytest.raises(ValueError, match="nesting violation"):
        sel.mark_rejected(np.asarray([False, True, True, False]))
    # Monotone growth is fine.
    sel.mark_rejected(np.asarray([True, False, True, True]))
    assert sel.rejected.tolist() == [True, False, True, True]


def test_selector_fail_closed_inputs() -> None:
    with pytest.raises(ValueError):
        EPsSelector(0)
    with pytest.raises(ValueError):
        EPsSelector(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EPsSelector(3, policy="ucb")
    with pytest.raises(ValueError):
        EPsSelector(3, seed=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        EPsSelector(3, variance_inflation=0.0)
    with pytest.raises(ValueError):
        EPsSelector(3, initial_variance=-1.0)
    sel = EPsSelector(3, seed=SEED)
    with pytest.raises(ValueError):
        sel.observe(3, 0.1)
    with pytest.raises(ValueError):
        sel.observe(-1, 0.1)
    with pytest.raises(ValueError):
        sel.observe(0, float("nan"))
    with pytest.raises(ValueError):
        sel.observe(0, float("inf"))
    with pytest.raises(ValueError):
        sel.mark_rejected(np.asarray([True, False]))
    sel.mark_rejected(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="must stop"):
        sel.select()


def test_selector_policies_all_run() -> None:
    world = _svs_world()
    for policy in POLICIES:
        res = eps_run(
            make_oracle(world),
            world.n_hyp,
            ALPHA,
            400,
            seed=SEED,
            policy=policy,
            nonnull_mask=world.nonnull,
            stop_rule=_discovery_rule(world.nonnull),
        )
        assert res.stopped_by in ("stop_rule", "budget")
        assert res.total_samples > 0


# ------------------------------------------------------------- oracle ----


def test_lr_oracle_matches_closed_form() -> None:
    world = _svs_world(n_hyp=5, k_nonnull=2)
    oracle = make_oracle(world)
    rng = np.random.default_rng(5)
    z = oracle(2, rng)
    rng2 = np.random.default_rng(5)
    y = rng2.normal(world.theta_true[2], np.sqrt(world.var[2]))
    theta = world.theta_test[2]
    expected = theta * y / world.var[2] - theta**2 / (2.0 * world.var[2])
    assert z == pytest.approx(expected)
    # Remark 2 sub-Gaussian proxy sigma_k^2 = theta^2 / v.
    assert oracle.variance_fn(2, 1) == pytest.approx(theta**2 / world.var[2])  # type: ignore[attr-defined]


def test_composite_null_truths_lie_inside_null_set() -> None:
    world = _cvs_world()
    nulls = ~world.nonnull
    assert bool(np.all(world.theta_true[nulls] <= world.theta_boundary[nulls]))
    assert bool(np.all(world.theta_true[nulls] >= -0.3))
    assert bool(np.all(world.theta_true[world.nonnull] > 0.0))


def test_plugin_oracle_first_increment_zero_and_predictable() -> None:
    world = _svc_world(n_hyp=4, k_nonnull=2)
    oracle = make_oracle(world)
    rng = np.random.default_rng(9)
    # theta-hat_{k,0} = theta-null => Q-hat_0 = P-null => log e = 0 exactly.
    assert oracle(1, rng) == 0.0
    # Second pull uses the running mean of the FIRST observation only
    # (predictability of the plug-in, their Eq. 17).
    rng_a = np.random.default_rng(11)
    y1 = rng_a.normal(world.theta_true[1], np.sqrt(world.var[1]))
    y2 = rng_a.normal(world.theta_true[1], np.sqrt(world.var[1]))
    oracle_b = make_oracle(world)
    rng_b = np.random.default_rng(11)
    oracle_b(1, rng_b)
    z2 = oracle_b(1, rng_b)
    v = world.var[1]
    expected = (-((y2 - y1) ** 2) + (y2 - 0.0) ** 2) / (2 * v)  # theta-hat_1 = y1
    assert z2 == pytest.approx(expected)


# ------------------------------------------------------------- eps_run ----


def test_eps_run_determinism() -> None:
    world = _svc_world()  # stateful plug-in oracle: strictest determinism test
    res_a = eps_run(
        make_oracle(world), world.n_hyp, ALPHA, 300, seed=SEED, nonnull_mask=world.nonnull
    )
    res_b = eps_run(
        make_oracle(world), world.n_hyp, ALPHA, 300, seed=SEED, nonnull_mask=world.nonnull
    )
    assert np.array_equal(res_a.arms, res_b.arms)
    assert np.array_equal(res_a.log_increments, res_b.log_increments)
    assert np.array_equal(res_a.rejected, res_b.rejected)
    assert res_a.discovery_time == res_b.discovery_time
    # Different seed -> (almost surely) a different sampling path.
    res_c = eps_run(
        make_oracle(world), world.n_hyp, ALPHA, 300, seed=SEED + 1, nonnull_mask=world.nonnull
    )
    assert not np.array_equal(res_a.arms, res_c.arms)


def test_eps_run_oracle_nan_fails_closed() -> None:
    def bad_oracle(arm: int, rng: np.random.Generator) -> float:
        return float("nan") if arm == 1 else 0.0

    with pytest.raises(ValueError, match="non-finite log increment"):
        eps_run(bad_oracle, 3, ALPHA, 50, seed=SEED)


def test_eps_run_fail_closed_inputs() -> None:
    world = _svs_world(n_hyp=4, k_nonnull=1)
    oracle = make_oracle(world)
    with pytest.raises(ValueError):
        eps_run(oracle, 4, 0.0, 10, seed=SEED)
    with pytest.raises(ValueError):
        eps_run(oracle, 4, 1.0, 10, seed=SEED)
    with pytest.raises(ValueError):
        eps_run(oracle, 0, ALPHA, 10, seed=SEED)
    with pytest.raises(ValueError):
        eps_run(oracle, 4, ALPHA, 0, seed=SEED)
    with pytest.raises(ValueError):
        eps_run(oracle, 4, ALPHA, 10, seed=SEED, policy="greedy")
    with pytest.raises(ValueError):
        eps_run(np.asarray([1.0]), 4, ALPHA, 10, seed=SEED)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        eps_run(oracle, 4, ALPHA, 10, seed=SEED, nonnull_mask=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError):
        eps_run(oracle, 4, ALPHA, 10, seed=SEED, nonnull_mask=np.ones(5, dtype=bool))


def test_eps_run_stop_rule_snapshot_is_a_copy() -> None:
    world = _svs_world(n_hyp=6, k_nonnull=2)
    captured: list[StepSnapshot] = []

    def rule(snap: StepSnapshot) -> bool:
        captured.append(snap)
        snap.log_e[:] = 1e9  # mutation must not leak into the run
        return snap.t >= 30

    res = eps_run(
        make_oracle(world),
        world.n_hyp,
        ALPHA,
        100,
        seed=SEED,
        nonnull_mask=world.nonnull,
        stop_rule=rule,
    )
    assert res.stopped_by == "stop_rule"
    assert res.total_samples == 30
    assert float(np.max(res.log_e)) < 1e8
    assert len(captured) == 30


def test_eps_run_all_rejected_break() -> None:
    # Every hypothesis nonnull: R_t = [K] eventually fires Algorithm 1 line 13.
    world = make_simple_vs_simple_world(4, 4, effect=1.5, stagger=0.0, var=1.0, seed=SEED)
    res = eps_run(
        make_oracle(world), world.n_hyp, ALPHA, 2000, seed=SEED, nonnull_mask=world.nonnull
    )
    assert res.stopped_by == "all_rejected"
    assert res.full_rejection_time == res.total_samples
    assert bool(np.all(res.rejected))


def test_eps_run_discovery_time_matches_history() -> None:
    world = _svs_world()
    res = eps_run(
        make_oracle(world),
        world.n_hyp,
        ALPHA,
        800,
        seed=SEED,
        nonnull_mask=world.nonnull,
        record_history=True,
    )
    assert res.discovery_time is not None
    first = next(
        t for t, mask in enumerate(res.history, start=1) if bool(np.all(mask[world.nonnull]))
    )
    assert first == res.discovery_time
    assert len(res.history) == res.total_samples
    # tau_* bookkeeping: TPR is complete at the discovery stop.
    assert res.tpr_final == pytest.approx(1.0)


@pytest.mark.parametrize("factory", _WORLD_FACTORIES)
def test_nested_rejection_sets_on_seeded_runs(factory) -> None:
    # Proposition 3.1: R_{t-1} subset R_t for every t (discoveries never
    # revoked), asserted step-by-step on seeded runs of all specializations.
    world = factory()
    res = eps_run(
        make_oracle(world),
        world.n_hyp,
        ALPHA,
        600,
        seed=SEED,
        nonnull_mask=world.nonnull,
        record_history=True,
        stop_rule=_discovery_rule(world.nonnull),
    )
    prev = np.zeros(world.n_hyp, dtype=bool)
    for mask in res.history:
        assert not bool(np.any(prev & ~mask)), "rejection set shrank — nesting violated"
        prev = mask
    assert res.discovery_time is not None  # budget is generous on these worlds
    assert bool(np.all(prev[world.nonnull]))  # stopped exactly at full discovery


# ------------------------------------------------- fixed-design baseline ----


def test_fixed_design_ebh_discovers_and_censors() -> None:
    world = _svs_world(n_hyp=8, k_nonnull=2, seed=SEED + 3)
    fd = fixed_design_ebh(
        make_oracle(world),
        world.n_hyp,
        ALPHA,
        200,
        seed=SEED,
        nonnull_mask=world.nonnull,
    )
    assert fd.discovery_n is not None
    assert fd.n_per_arm == fd.discovery_n
    assert fd.total_samples == fd.discovery_n * world.n_hyp
    assert bool(np.all(fd.rejected[world.nonnull]))
    assert fd.tpr_final == pytest.approx(1.0)
    assert 0.0 <= fd.fdp_final <= 1.0
    # Censoring: impossible budget -> discovery_n None, full n_per_arm used.
    fd2 = fixed_design_ebh(
        make_oracle(world), world.n_hyp, ALPHA, 2, seed=SEED, nonnull_mask=world.nonnull
    )
    assert fd2.discovery_n is None
    assert fd2.n_per_arm == 2
    assert fd2.total_samples == 2 * world.n_hyp
    with pytest.raises(ValueError):
        fixed_design_ebh(make_oracle(world), 8, ALPHA, 0, seed=SEED)


# ------------------------------------------- anytime-valid FDR (MC) ----


@pytest.mark.parametrize(
    "rule_name",
    ["fixed_time", "first_rejection", "uniform_random_time", "discovery_time"],
)
def test_anytime_fdr_at_stopping_rules(rule_name: str) -> None:
    # Definition 2.1: FDR(R_tau) <= alpha at ANY stopping time tau w.r.t.
    # {F_t}, including data-dependent ones. The planted H1 is deterministic
    # given the world, so tau_*-style rules are F_t-measurable.
    world = _svs_world(n_hyp=10, k_nonnull=3, seed=SEED + 4)
    reps, budget = 100, 250
    fdps: list[float] = []
    for i in range(reps):
        if rule_name == "fixed_time":
            rule = lambda snap: snap.t >= 60  # noqa: E731
        elif rule_name == "first_rejection":
            rule = lambda snap: bool(np.any(snap.rejected))  # noqa: E731
        elif rule_name == "uniform_random_time":
            tau = int(np.random.default_rng(10_000 + i).integers(1, 151))
            rule = lambda snap, tau=tau: snap.t >= tau  # noqa: E731
        else:
            rule = _discovery_rule(world.nonnull)
        res = eps_run(
            make_oracle(world),
            world.n_hyp,
            ALPHA,
            budget,
            seed=SEED + 7 * i,
            nonnull_mask=world.nonnull,
            stop_rule=rule,
        )
        fdps.append(res.fdp_final)
    fdr = float(np.mean(fdps))
    assert fdr <= ALPHA + 0.03, f"FDR {fdr:.4f} exceeds alpha={ALPHA} under rule {rule_name}"


@pytest.mark.parametrize("factory", [_svs_world, _svc_world])
def test_anytime_fdr_global_null(factory) -> None:
    # Global null: FDR = P(|R_tau| >= 1). Both the LR increments (Eq. 12) and
    # the plug-in increments (Eq. 17) must keep e-BH honest while sampling
    # is adaptive and the stop is data-dependent (first rejection).
    world = factory(n_hyp=10, k_nonnull=0, seed=SEED + 5)
    assert not bool(np.any(world.nonnull))
    reps, budget = 80, 150
    hits = 0
    for i in range(reps):
        res = eps_run(
            make_oracle(world),
            world.n_hyp,
            ALPHA,
            budget,
            seed=SEED + 11 * i,
            stop_rule=lambda snap: bool(np.any(snap.rejected)),
        )
        hits += int(res.num_rejected > 0)
    fdr = hits / reps
    assert fdr <= ALPHA + 0.02, f"global-null FDR {fdr:.4f} exceeds alpha={ALPHA}"


# --------------------------------------------- efficiency (headline) ----


@pytest.mark.parametrize("factory", _WORLD_FACTORIES)
def test_eps_beats_round_robin_and_fixed_design(factory) -> None:
    # Headline comparison: on seeded SYNTHETIC planted worlds, adaptive e-PS
    # discovers all nonnulls with FEWER total samples than (a) uniform
    # round-robin allocation + e-BH and (b) fixed-design e-BH at matched
    # alpha (paper Sections 1 and 6, Figure 1-2).
    world = factory()
    bench = bench_eps_efficiency(world, ALPHA, n_seeds=8, seed=SEED, budget=3000, n_per_arm_max=150)
    assert bench["synthetic"] is True
    assert bench["eps_censored"] == 0
    eps_mean = float(bench["eps_discovery_samples_mean"])  # type: ignore[arg-type]
    rr_mean = float(bench["round_robin_discovery_samples_mean"])  # type: ignore[arg-type]
    fixed_mean = float(bench["fixed_design_discovery_samples_mean"])  # type: ignore[arg-type]
    assert np.isfinite(eps_mean) and np.isfinite(rr_mean) and np.isfinite(fixed_mean)
    assert eps_mean < rr_mean
    assert eps_mean < fixed_mean
    assert float(bench["eps_speedup_vs_round_robin"]) > 1.1  # type: ignore[arg-type]
    assert float(bench["eps_speedup_vs_fixed_design"]) > 1.1  # type: ignore[arg-type]
    # FDR stays controlled at the discovery stop (mean FDP <= alpha + slack).
    assert float(bench["eps_fdp_at_discovery_mean"]) <= ALPHA + 0.05  # type: ignore[arg-type]
    assert float(bench["eps_tpr_mean"]) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_bench_keys_and_fail_closed() -> None:
    world = _svs_world(n_hyp=8, k_nonnull=2, seed=SEED + 6)
    bench = bench_eps_efficiency(
        world,
        ALPHA,
        n_seeds=3,
        seed=SEED,
        budget=2000,
        n_per_arm_max=100,
        policies=("eps", "round_robin", "uniform"),
    )
    required = {
        "label",
        "synthetic",
        "specialization",
        "n_hyp",
        "k_nonnull",
        "alpha",
        "n_seeds",
        "budget",
        "n_per_arm_max",
        "eps_discovery_samples_mean",
        "eps_discovery_samples_max",
        "eps_censored",
        "eps_fdp_at_discovery_mean",
        "eps_tpr_mean",
        "round_robin_discovery_samples_mean",
        "round_robin_censored",
        "uniform_discovery_samples_mean",
        "uniform_censored",
        "fixed_design_discovery_samples_mean",
        "fixed_design_censored",
        "fixed_design_fdp_at_discovery_mean",
        "fixed_design_tpr_mean",
        "eps_speedup_vs_round_robin",
        "eps_speedup_vs_fixed_design",
    }
    assert required <= set(bench.keys())
    for key, value in bench.items():
        assert isinstance(value, (bool, int, float, str)), f"non-scalar bench value {key}"
    assert bench["specialization"] == "simple_vs_simple"
    assert bench["k_nonnull"] == 2
    with pytest.raises(ValueError):
        bench_eps_efficiency(world, 1.5)
    with pytest.raises(ValueError):
        bench_eps_efficiency(world, ALPHA, policies=())
    with pytest.raises(ValueError):
        bench_eps_efficiency(world, ALPHA, policies=("greedy",))
    with pytest.raises(ValueError):
        bench_eps_efficiency(world, ALPHA, n_seeds=0)
    with pytest.raises(ValueError):
        bench_eps_efficiency(world, ALPHA, budget=1)
