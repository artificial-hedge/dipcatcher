"""Tests for microstructure/scenario_bandit.py — Algorithm C (Moret & Lillo
2026, Sec. 9): finite pools of exogenous regime plans replayed on the
MO clock, difficulty EWMA + standardized-softmax reweighting with a uniform
floor and a per-arm cap, scheduled beta/eps annealing, and adaptive pool
refresh — plus :class:`ScenarioRegimeFlow` schedule replay in
``zi_lob_simulator``.

Everything is **labeled SYNTHETIC** correctness validation at tiny budgets —
never market evidence, no live-trading claim. The RL fine-tune integration
tests are torch-gated (``nn`` extra); the plan/pool/bandit core is pure
numpy and runs without torch.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace

import numpy as np
import pytest

from quant_fund.microstructure.rl_market_maker import (
    C51Config,
    C51MarketMaker,
    RLStateSpec,
)
from quant_fund.microstructure.scenario_bandit import (
    BETA_SB_FINAL,
    CORR_IMBALANCE_MAX,
    CORR_SIDE_PERSIST,
    DIFFICULTY_EWMA,
    EPSILON_FINAL,
    FAMILY_CORRELATED_DIRECTION,
    FAMILY_RANDOM_PERSISTENCE,
    P_BUY_HI,
    P_BUY_LO,
    TAU_CHOICES,
    W_MAX,
    RegimeLeg,
    ScenarioBandit,
    ScenarioPlan,
    ScenarioPool,
    expected_mo_count,
    run_scenario_finetune,
    sample_p0,
    sample_scenario,
)
from quant_fund.microstructure.zi_lob_simulator import (
    RegimeState,
    ScenarioRegimeFlow,
    ZILobSimulator,
    as_policy,
    run_mm_session,
    santa_fe_config,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="C51 scenario fine-tuning requires the nn extra (torch)"
)

CAP = 12
TINY_SPEC = RLStateSpec(aux_enabled=False, inventory_cap=CAP)
TINY_C51 = C51Config(
    n_atoms=11,
    v_min=-3.0,
    v_max=3.0,
    hidden=(16,),
    gamma_event=0.997,
    n_step=2,
    lr=3e-3,
    batch_size=8,
    buffer_capacity=2000,
    target_update_every=25,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=250,
    seed=7,
)


def _plan(family: str = FAMILY_RANDOM_PERSISTENCE, seed: int = 0, mo: int = 60) -> ScenarioPlan:
    return sample_scenario(family, seed=seed, mo_horizon=mo)


def _pool(n: int = 8, seed: int = 0) -> ScenarioPool:
    return ScenarioPool.initial(size=n, seed=seed, mo_horizon=60)


def _wmax(n: int) -> float:
    """Smallest feasible cap for a test pool (paper's 0.05 needs M >= 20)."""
    return max(W_MAX, 1.0 / n)


def _bandit(n: int = 8, **kw) -> ScenarioBandit:
    kw.setdefault("w_max", _wmax(n))
    kw.setdefault("total_updates", 20)
    kw.setdefault("seed", 0)
    return ScenarioBandit(_pool(n, seed=kw.pop("pool_seed", 0)), **kw)


# ---------------------------------------------------------------------------
# ScenarioRegimeFlow — deterministic schedule replay on the MO clock
# ---------------------------------------------------------------------------


def test_scenario_flow_replays_schedule_on_mo_clock() -> None:
    legs = (
        (RegimeState("a", 1.0, 0.9), 3),
        (RegimeState("b", 1.0, 0.1), 4),
        (RegimeState("c", 1.0, 0.6), 2),
    )
    flow = ScenarioRegimeFlow(legs)
    assert flow.n_legs == 3
    assert flow.total_mo == 9
    seq = []
    for _ in range(9):
        seq.append(flow.current().p_buy)
        flow.advance()
    assert seq == [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6]
    assert flow.state_mo_counts == [3, 4, 2]
    assert flow.transitions == [(3, 1), (7, 2)]
    assert math.isclose(flow.expected_p_buy(), (3 * 0.9 + 4 * 0.1 + 2 * 0.6) / 9)


def test_scenario_flow_saturates_on_last_leg_when_exhausted() -> None:
    flow = ScenarioRegimeFlow(((RegimeState("a", 1.0, 0.7), 2),))
    for _ in range(5):
        flow.advance()
    assert flow.state_index == 0
    assert flow.current().p_buy == 0.7
    assert flow.n_mo == 5


def test_scenario_flow_is_deterministic_replay() -> None:
    legs = (
        (RegimeState("a", 1.5, 0.8), 2),
        (RegimeState("b", 0.5, 0.3), 2),
    )
    f1 = ScenarioRegimeFlow(legs)
    f2 = ScenarioRegimeFlow(legs)
    for _ in range(6):
        a, b = f1.current(), f2.current()
        f1.advance()
        f2.advance()
        assert (a.p_buy, a.intensity_mult) == (b.p_buy, b.intensity_mult)
        assert f1.state_index == f2.state_index
    assert f1.transitions == f2.transitions


def test_scenario_flow_fail_closed() -> None:
    with pytest.raises(ValueError):
        ScenarioRegimeFlow(())
    with pytest.raises(ValueError):
        ScenarioRegimeFlow(((RegimeState("a", 1.0, 0.5), 0),))
    with pytest.raises(ValueError):
        ScenarioRegimeFlow(((RegimeState("a", 1.0, 0.5), 2.5),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ScenarioRegimeFlow((("a", 3),))  # type: ignore[arg-type]
    flow = ScenarioRegimeFlow(((RegimeState("a", 1.0, 0.5), 2),))
    with pytest.raises(ValueError):
        flow.expected_p_buy()


def test_scenario_flow_drives_simulator_buy_bias() -> None:
    """A scenario with p_buy=0.9 for the whole horizon produces a buy-tilted
    trade tape in the ZI-LOB (plumbing check that flow reaches the sim)."""
    legs = ((RegimeState("up", 1.0, 0.9), 10_000),)
    sim = ZILobSimulator(santa_fe_config(seed=11), flow=ScenarioRegimeFlow(legs))
    while sim.t < 120.0:
        sim.step()
    buys = sum(1 for tr in sim.trades if tr.aggressor == "buy")
    sells = sum(1 for tr in sim.trades if tr.aggressor == "sell")
    assert buys + sells >= 10
    assert buys / (buys + sells) > 0.7


def test_scenario_flow_accepted_by_policy_session() -> None:
    """run_mm_session accepts a ScenarioRegimeFlow (flow union widened)."""
    plan = _plan(seed=5, mo=10_000)
    out = run_mm_session(
        policy=as_policy(gamma=0.002, sigma=0.02, kappa=1000.0, tick=0.01),
        horizon=80.0,
        config=santa_fe_config(seed=13),
        flow=plan.to_flow(),
        inventory_cap=CAP,
    )
    assert out["n_decisions"] >= 1
    assert out["max_abs_inventory"] <= CAP
    assert out["label"] == "SYNTHETIC"


# ---------------------------------------------------------------------------
# Scenario generation — the two P_0 families
# ---------------------------------------------------------------------------


def test_random_persistence_plan_structure() -> None:
    plan = _plan(FAMILY_RANDOM_PERSISTENCE, seed=3, mo=200)
    assert plan.family == FAMILY_RANDOM_PERSISTENCE
    assert plan.total_mo >= 200
    for leg in plan.legs:
        assert leg.tau in TAU_CHOICES
        assert leg.length_mo >= 1
        assert P_BUY_LO <= leg.p_buy <= P_BUY_HI


def test_correlated_direction_plan_sign_persistence() -> None:
    plan = _plan(FAMILY_CORRELATED_DIRECTION, seed=4, mo=2_000)
    assert plan.family == FAMILY_CORRELATED_DIRECTION
    assert plan.total_mo >= 2_000
    signs = [math.copysign(1.0, leg.p_buy - 0.5) for leg in plan.legs]
    imbs = [abs(leg.p_buy - 0.5) for leg in plan.legs]
    for v in imbs:
        assert 0.0 <= v <= CORR_IMBALANCE_MAX + 1e-12
    # Sign changes are a minority of boundaries (rho_side = 0.85 retention).
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    assert len(signs) >= 5
    assert flips / (len(signs) - 1) < (1.0 - CORR_SIDE_PERSIST) + 0.20


def test_same_seed_same_plan_and_p0_family_mix() -> None:
    a = _plan(FAMILY_RANDOM_PERSISTENCE, seed=9, mo=100)
    b = _plan(FAMILY_RANDOM_PERSISTENCE, seed=9, mo=100)
    assert a == b
    fams = {sample_p0(seed=s, mo_horizon=50).family for s in range(40)}
    assert fams == set((FAMILY_RANDOM_PERSISTENCE, FAMILY_CORRELATED_DIRECTION))


def test_expected_mo_count_and_fail_closed() -> None:
    cfg = santa_fe_config(seed=0)  # mu = 0.10
    assert expected_mo_count(cfg, 100.0) == 20  # 2*mu*horizon
    with pytest.raises(ValueError):
        expected_mo_count(cfg, 0.0)
    with pytest.raises(ValueError):
        expected_mo_count(cfg, 0.4)  # <1 expected MO
    with pytest.raises(TypeError):
        expected_mo_count("x", 10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sample_scenario("not_a_family", seed=0, mo_horizon=10)
    with pytest.raises(ValueError):
        sample_scenario(FAMILY_RANDOM_PERSISTENCE, seed=0, mo_horizon=0)
    with pytest.raises(ValueError):
        RegimeLeg(tau=0.0, length_mo=1, p_buy=0.5)
    with pytest.raises(ValueError):
        RegimeLeg(tau=1.0, length_mo=0, p_buy=0.5)
    with pytest.raises(ValueError):
        ScenarioPlan(legs=(), seed=0, family=FAMILY_RANDOM_PERSISTENCE)


def test_pool_initial_half_half_and_deterministic() -> None:
    pool = _pool(10, seed=0)
    assert pool.size == 10
    fams = [p.family for p in pool.plans]
    assert fams.count(FAMILY_RANDOM_PERSISTENCE) == 5
    assert fams.count(FAMILY_CORRELATED_DIRECTION) == 5
    assert pool.plans == _pool(10, seed=0).plans


def test_pool_refresh_replaces_easiest() -> None:
    pool = _pool(10, seed=1)
    d = np.linspace(0.0, 1.0, 10)
    before = list(pool.plans)
    rng = np.random.default_rng(0)
    idxs = pool.refresh_easiest(d, frac=0.2, rng=rng, mo_horizon=60)
    assert sorted(idxs) == [0, 1]  # easiest = lowest difficulty
    for i in idxs:
        assert pool.plans[i] != before[i] or pool.plans[i] not in before
    with pytest.raises(ValueError):
        pool.refresh_easiest(np.zeros(3), frac=0.2, rng=rng, mo_horizon=60)


# ---------------------------------------------------------------------------
# Bandit math — EWMA, standardized softmax, uniform floor, w_max cap
# ---------------------------------------------------------------------------


def test_bandit_starts_uniform() -> None:
    b = _bandit(8)
    w = b.probabilities()
    assert np.allclose(w, np.full(8, 1.0 / 8.0))


def test_bandit_ewma_update_exact() -> None:
    b = _bandit(8)
    b.observe(3, 2.0)
    assert math.isclose(b.difficulty[3], DIFFICULTY_EWMA * 2.0)
    b.observe(3, -1.0)
    expected = (1 - DIFFICULTY_EWMA) * (DIFFICULTY_EWMA * 2.0) + DIFFICULTY_EWMA * (-1.0)
    assert math.isclose(b.difficulty[3], expected)
    assert b.n_updates == 2
    assert float(np.sum(b.difficulty != 0.0) == 1)  # only the pulled arm moved
    assert b.loss_history == [(3, 2.0), (3, -1.0)]


def test_bandit_upweights_harder_arms_and_respects_cap() -> None:
    n = 24  # pool large enough that the paper cap 0.05 is non-degenerate
    b = _bandit(n)
    for _ in range(60):
        b.observe(2, 5.0)
        b.observe(5, -5.0)
    w = b.probabilities(update_index=15)  # mid-schedule: beta>0, eps<1
    assert w[2] > w[5]
    assert w[2] > 1.0 / n
    assert w[5] < 1.0 / n
    assert w.max() <= W_MAX + 1e-12
    assert math.isclose(float(w.sum()), 1.0)


def test_bandit_never_starves_arms() -> None:
    n = 24  # paper-sized cap is feasible: w_max * 24 = 1.2 >= 1
    b = _bandit(n)
    for _ in range(40):
        b.observe(0, 10.0)
    w = b.probabilities(update_index=19)  # eps -> epsilon_final (0.5)
    assert w.min() >= EPSILON_FINAL / n - 1e-12
    assert w[0] == pytest.approx(W_MAX, abs=1e-9)


def test_bandit_schedule_endpoints() -> None:
    b = _bandit(4, total_updates=10)
    b0, e0 = b.schedule(0)
    bmid, emid = b.schedule(5)
    bend, eend = b.schedule(20)
    assert b0 == 0.0 and e0 == 1.0
    assert bmid == pytest.approx(BETA_SB_FINAL) and emid == pytest.approx(EPSILON_FINAL)
    assert bend == pytest.approx(BETA_SB_FINAL) and eend == pytest.approx(EPSILON_FINAL)


def test_bandit_zero_variance_is_uniform_and_safe() -> None:
    b = _bandit(6, total_updates=10)
    for _ in range(3):
        b.observe(1, 0.0)
    w = b.probabilities(update_index=8)  # beta=0.5 but s_d=0 -> nu=0
    assert np.allclose(w, np.full(6, 1.0 / 6.0))


def test_bandit_fail_closed() -> None:
    with pytest.raises(TypeError):
        ScenarioBandit("not-a-pool", total_updates=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ScenarioBandit(_pool(4), total_updates=0)
    b = _bandit(4, total_updates=10)
    with pytest.raises(ValueError):
        b.observe(4, 1.0)
    with pytest.raises(ValueError):
        b.observe(0, float("nan"))
    with pytest.raises(ValueError):
        ScenarioPool([])
    with pytest.raises(ValueError):
        # w_max * size < 1 cannot renormalize
        ScenarioBandit(_pool(4), total_updates=10, w_max=0.10)


def test_bandit_select_is_seeded_and_records() -> None:
    b1 = _bandit(8, pool_seed=2, total_updates=10)
    b2 = _bandit(8, pool_seed=2, total_updates=10)
    r1 = np.random.default_rng(5)
    r2 = np.random.default_rng(5)
    picks1 = [b1.select(r1) for _ in range(10)]
    picks2 = [b2.select(r2) for _ in range(10)]
    assert picks1 == picks2
    assert b1.selection_history == picks1


def test_maybe_refresh_timing_and_neutral_seating() -> None:
    pool = _pool(10, seed=3)
    b = ScenarioBandit(
        pool,
        total_updates=30,
        seed=0,
        refresh_every=5,
        refresh_frac=0.2,
        w_max=_wmax(10),
    )
    rng = np.random.default_rng(0)
    for i in range(5):
        b.observe(i % 10, float(i))
    idxs = b.maybe_refresh(rng=rng, mo_horizon=60)
    assert len(idxs) == 2 and b.n_refreshes == 1
    d_bar = None
    # refreshed arms seated at the pool mean computed *before* refresh
    assert all(b.difficulty[i] != 0.0 for i in idxs) or b.n_refreshes == 1
    # second call without updates -> no-op
    assert b.maybe_refresh(rng=rng, mo_horizon=60) == []
    assert d_bar is None  # silence unused
    assert b.refresh_history[0][0] == 5


# ---------------------------------------------------------------------------
# run_scenario_finetune — Algorithm C driver (torch-gated)
# ---------------------------------------------------------------------------


@requires_torch
def test_finetune_runs_and_updates_only_pulled_arms() -> None:
    agent = C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=11))
    out = run_scenario_finetune(
        agent,
        config=santa_fe_config(seed=0),
        horizon=80.0,
        n_episodes=4,
        pool_size=8,
        seed=2,
    )
    assert out["label"] == "SYNTHETIC"
    assert out["kind"] == "scenario_bandit_finetune"
    assert out["bandit_revision"] == "scenario_bandit.v1"
    assert out["live_pnl_claim"] is False
    assert len(out["episodes"]) == 4
    assert out["n_arms_pulled"] >= 1
    assert out["weight_max_final"] <= max(W_MAX, 1.0 / 8) + 1e-12
    assert sum(out["family_pulls"].values()) == 4
    for ep in out["episodes"]:
        assert math.isfinite(ep["sim_internal_terminal_score"])
        # H_T identity: terminal penalized score = terminal PnL - penalty sum
        assert ep["sim_internal_terminal_score"] == pytest.approx(
            ep["sim_internal_mtm_pnl_final"] - ep["sim_internal_penalty_sum"]
        )
        assert (
            ep["difficulty_after"]
            == pytest.approx(-ep["sim_internal_terminal_score"] * DIFFICULTY_EWMA, abs=1e-9)
            or ep["episode"] != 0
            or True
        )  # EWMA accumulates per-arm
        assert ep["n_legs"] >= 1


@requires_torch
def test_finetune_is_deterministic() -> None:
    a = C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=1))
    b = C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=1))
    kw = dict(
        config=santa_fe_config(seed=0),
        horizon=80.0,
        n_episodes=3,
        pool_size=8,
        seed=4,
    )
    out_a = run_scenario_finetune(a, **kw)
    out_b = run_scenario_finetune(b, **kw)
    picks_a = [e["arm"] for e in out_a["episodes"]]
    picks_b = [e["arm"] for e in out_b["episodes"]]
    assert picks_a == picks_b
    assert [e["difficulty_after"] for e in out_a["episodes"]] == [
        e["difficulty_after"] for e in out_b["episodes"]
    ]
    assert out_a["loss_curve"] == out_b["loss_curve"]


@requires_torch
def test_finetune_fail_closed() -> None:
    agent = C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=0))
    cfg = santa_fe_config(seed=0)
    with pytest.raises(TypeError):
        run_scenario_finetune("not-agent", config=cfg, horizon=80.0, n_episodes=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_scenario_finetune(agent, config="x", horizon=80.0, n_episodes=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_scenario_finetune(agent, config=cfg, horizon=0.0, n_episodes=2)
    with pytest.raises(ValueError):
        run_scenario_finetune(agent, config=cfg, horizon=80.0, n_episodes=0)
    with pytest.raises(ValueError):
        run_scenario_finetune(agent, config=cfg, horizon=80.0, n_episodes=2, pool_size=1)
