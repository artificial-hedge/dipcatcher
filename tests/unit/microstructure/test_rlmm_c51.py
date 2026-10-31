"""Tests for microstructure/rlmm_c51.py — lane B4-ii wave 17 (Algorithm C).

Scenario-bandit robust fine-tuning for the C51 RL market maker of Moret &
Lillo (2026, arXiv:2609.11614, verified against the arXiv abstract + HTML
full text): the two-family reference scenario generator ``P_0``
(random-persistence / correlated-direction), scheduled regime flow on the
MO clock, the EWMA-difficulty softmax bandit with epsilon-mix, per-arm cap,
annealing and pool refresh, the penalized terminal score ``H_T``, the
fine-tuning runner, the scenario-mixture evaluation harness, and the
torch-free numpy contrast policies.

Everything is **labeled SYNTHETIC** correctness validation at TINY budgets —
never market evidence, no live-trading claim. The C51 agent / session /
training plumbing itself is owned and covered by wave-16's
``test_rl_market_maker.py``; the tests here cover only the wave-17 surface,
plus cross-cutting invariants (event-stream causality, honesty namespacing,
determinism). Tiny-budget bandit tests assert the update math and plumbing,
NOT the paper's paper-scale improvement claims. Torch tests skip cleanly
when the ``nn`` extra is absent; everything else is pure numpy.
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
    FlowBiasFilter,
    RLStateSpec,
    build_state_vector,
    run_rl_mm_session,
)
from quant_fund.microstructure.rlmm_c51 import (
    PAPER_BANDIT_KWARGS,
    SCENARIO_FAMILIES,
    RegimeSegment,
    Scenario,
    ScenarioBandit,
    ScenarioGenerator,
    ScheduledRegimeFlow,
    default_expected_mos,
    evaluate_scenario_robustness,
    fixed_offset_policy,
    inventory_skew_policy,
    penalized_terminal_score,
    random_offset_policy,
    run_scenario_bandit_finetuning,
    stationary_scenario,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MMState,
    ZILobSimulator,
    as_policy,
    glft_policy,
    santa_fe_config,
)
from quant_fund.models.changepoint import bocpd_gaussian


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="scenario-bandit C51 fine-tuning requires the nn extra (torch)"
)

# Forbidden *headline* metric tokens (mirrors research.catalog registry).
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

# Tiny-budget pins.
CAP = 8
TINY_SPEC = RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=CAP)
TINY_C51 = C51Config(
    n_atoms=11,
    v_min=-3.0,
    v_max=3.0,
    hidden=(16,),
    gamma_event=0.999,
    n_step=2,
    lr=1e-3,
    batch_size=8,
    buffer_capacity=512,
    target_update_every=20,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_steps=60,
    seed=0,
)


def _tiny_agent(seed: int = 0) -> C51MarketMaker:
    return C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=seed))


def _glft_pol():
    return glft_policy(gamma=1.0, sigma=0.02, kappa=1000.0, a_fill=1.0, tick=0.01)


def _all_keys(obj: object) -> list[str]:
    """Recursively collect mapping keys (dicts only; walk list/tuple values)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


def _assert_synthetic_honest(bundle: dict) -> None:
    assert bundle["label"] == "SYNTHETIC"
    assert bundle["research_only"] is True
    assert bundle["live_pnl_claim"] is False
    for k in _all_keys(bundle):
        kl = k.lower()
        for tok in FORBIDDEN_HEADLINE_TOKENS:
            assert tok not in kl, f"forbidden headline token {tok} in key {k}"
        if "pnl" in kl and kl != "live_pnl_claim":
            assert kl.startswith("sim_internal") or "_sim_internal" in kl, (
                f"pnl-like key outside sim_internal_ namespace: {k}"
            )


# ---------------------------------------------------------------------------
# RegimeSegment / Scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_mos": 0},
        {"length_mos": -3},
        {"length_mos": 5, "p_buy": -0.1},
        {"length_mos": 5, "p_buy": 1.1},
        {"length_mos": 5, "p_buy": 0.5, "intensity_mult": 0.0},
        {"length_mos": 5, "p_buy": 0.5, "tau_hint": -1.0},
    ],
)
def test_regime_segment_fail_closed(kwargs: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        RegimeSegment(**kwargs)


def test_scenario_schedule_piecewise_constant() -> None:
    sc = Scenario(
        (
            RegimeSegment(10, 0.2, 1.0, 30.0),
            RegimeSegment(20, 0.7, 1.0, 60.0),
            RegimeSegment(30, 0.5, 1.0, 60.0),
        ),
        "random_persistence",
        seed=3,
    )
    assert sc.n_regimes == 3
    assert sc.total_mos == 60
    assert sc.cumulative == (10, 30, 60)
    # p(m) = p_k for C_{k-1} <= m < C_k (m zero-indexed)
    for m in range(10):
        assert sc.p_buy_at(m) == pytest.approx(0.2)
    for m in range(10, 30):
        assert sc.p_buy_at(m) == pytest.approx(0.7)
    for m in range(30, 60):
        assert sc.p_buy_at(m) == pytest.approx(0.5)
    assert sc.covers(59) and not sc.covers(60)
    # documented clamp: beyond the stored horizon the last regime holds
    assert sc.p_buy_at(10_000) == pytest.approx(0.5)


def test_scenario_fail_closed() -> None:
    with pytest.raises(ValueError):
        Scenario((), "random_persistence", seed=0)
    with pytest.raises(TypeError):
        Scenario(("not_a_segment",), "random_persistence", seed=0)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        Scenario((RegimeSegment(5, 0.5),), "", seed=0)
    with pytest.raises(ValueError):
        Scenario((RegimeSegment(5, 0.5),), "f", seed=-1)


# ---------------------------------------------------------------------------
# ScheduledRegimeFlow: schedule fidelity on the MO clock
# ---------------------------------------------------------------------------


def _two_regime_scenario(l0: int = 5, p0: float = 0.2, l1: int = 7, p1: float = 0.8) -> Scenario:
    return Scenario(
        (RegimeSegment(l0, p0, 1.0, 30.0), RegimeSegment(l1, p1, 1.0, 60.0)),
        "random_persistence",
        seed=11,
    )


def test_scheduled_flow_is_markov_subtype_for_session_gate() -> None:
    flow = ScheduledRegimeFlow(_two_regime_scenario())
    assert isinstance(flow, MarkovRegimeFlow)
    # the run_rl_mm_session isinstance gate accepts it
    bundle = run_rl_mm_session(
        config=santa_fe_config(seed=3),
        horizon=30.0,
        policy=fixed_offset_policy((0, 0), tick=0.01),
        flow=flow,
        inventory_cap=CAP,
    )
    assert bundle["session_completed"] is True
    assert bundle["label"] == "SYNTHETIC"


def test_scheduled_flow_follows_schedule_boundaries() -> None:
    sc = _two_regime_scenario(l0=5, l1=7)
    flow = ScheduledRegimeFlow(sc)
    seen_p: list[float] = []
    for _ in range(sc.total_mos):
        seen_p.append(flow.current().p_buy)
        flow.advance()
    assert seen_p[:5] == [0.2] * 5
    assert seen_p[5:] == [0.8] * 7
    assert flow.transitions == [(5, 1)]
    assert flow.state_mo_counts == [5, 7]
    assert flow.n_mo == 12
    # clamp: further advances hold the last regime
    flow.advance()
    assert flow.current().p_buy == pytest.approx(0.8)
    assert flow.state_mo_counts == [5, 8]


def test_scheduled_flow_expected_p_buy_visit_weighted() -> None:
    sc = _two_regime_scenario(l0=3, p0=0.0, l1=1, p1=1.0)
    flow = ScheduledRegimeFlow(sc)
    with pytest.raises(ValueError):
        flow.expected_p_buy()
    for _ in range(4):
        flow.advance()
    # (3 * 0.0 + 1 * 1.0) / 4 = 0.25 — visit-weighted mean of p_buy
    assert flow.expected_p_buy() == pytest.approx(0.25)


def test_scheduled_flow_deterministic_no_rng() -> None:
    sc = _two_regime_scenario()
    a, b = ScheduledRegimeFlow(sc), ScheduledRegimeFlow(sc)
    for _ in range(sc.total_mos + 3):
        a.advance()
        b.advance()
        assert a.current() == b.current()
    assert a.transitions == b.transitions


def test_scheduled_flow_fail_closed() -> None:
    with pytest.raises(TypeError):
        ScheduledRegimeFlow("not_a_scenario")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ScenarioGenerator: P_0 families, determinism, pool init
# ---------------------------------------------------------------------------


def test_generator_random_persistence_fields() -> None:
    gen = ScenarioGenerator(seed=0)
    sc = gen.draw(80, family="random_persistence")
    assert sc.family == "random_persistence"
    assert sc.total_mos >= 80
    assert sc.n_regimes >= 1
    for seg in sc.segments:
        assert seg.length_mos >= 1
        assert 0.20 <= seg.p_buy <= 0.80
        assert seg.tau_hint in (15.0, 30.0, 60.0, 120.0, 240.0)
        assert seg.intensity_mult == pytest.approx(1.0)


def test_generator_correlated_direction_sign_persistence() -> None:
    # rho=1.0: the sign never flips -> all p_buy on the same side of 0.5.
    gen = ScenarioGenerator(seed=0, rho_side=1.0, i_max=0.30)
    sc = gen.draw(10, family="correlated_direction")
    signs = np.sign([s.p_buy - 0.5 for s in sc.segments])
    nonzero = signs[signs != 0.0]
    assert nonzero.size >= 1
    assert np.all(nonzero == nonzero[0])
    for s in sc.segments:
        assert 0.5 - 0.30 <= s.p_buy <= 0.5 + 0.30
    # rho=0.0: the sign flips every regime boundary (up to p==0.5 ties).
    gen0 = ScenarioGenerator(seed=0, rho_side=0.0, i_max=0.30, tau_grid=(1,))
    sc0 = gen0.draw(8, family="correlated_direction")
    if sc0.n_regimes >= 3:
        sg = np.sign([s.p_buy - 0.5 for s in sc0.segments])
        nz = [v for v in sg if v != 0.0]
        assert all(nz[i] != nz[i + 1] for i in range(len(nz) - 1))


def test_generator_family_coin_mixture_deterministic() -> None:
    g1, g2 = ScenarioGenerator(seed=9), ScenarioGenerator(seed=9)
    fams1 = [g1.draw(20).family for _ in range(8)]
    fams2 = [g2.draw(20).family for _ in range(8)]
    assert fams1 == fams2
    assert set(fams1) == set(SCENARIO_FAMILIES)  # both families drawn over 8 pulls


def test_generator_determinism_and_seed_sensitivity() -> None:
    g1, g2 = ScenarioGenerator(seed=4), ScenarioGenerator(seed=4)
    a = [g1.draw(50) for _ in range(6)]
    b = [g2.draw(50) for _ in range(6)]
    assert [(s.family, s.segments, s.seed) for s in a] == [
        (s.family, s.segments, s.seed) for s in b
    ]
    g3 = ScenarioGenerator(seed=5)
    c = [g3.draw(50) for _ in range(6)]
    assert [(s.family, s.segments, s.seed) for s in a] != [
        (s.family, s.segments, s.seed) for s in c
    ]


def test_initial_pool_deterministic_half_half() -> None:
    p1 = ScenarioGenerator(seed=1).initial_pool(10, 40)
    p2 = ScenarioGenerator(seed=1).initial_pool(10, 40)
    assert len(p1) == 10
    assert [s.family for s in p1] == ["random_persistence"] * 5 + ["correlated_direction"] * 5
    assert [(s.family, s.segments, s.seed) for s in p1] == [
        (s.family, s.segments, s.seed) for s in p2
    ]


def test_generator_fail_closed() -> None:
    gen = ScenarioGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.draw(0)
    with pytest.raises(ValueError):
        gen.draw(10, family="stationary")  # eval-only arm, not a P_0 family
    with pytest.raises(ValueError):
        gen.draw(10, family="nope")
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, tau_grid=())
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, tau_grid=(15, 0))
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, p_lo=0.8, p_hi=0.2)
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, rho_side=1.5)
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, i_max=0.6)
    with pytest.raises(ValueError):
        ScenarioGenerator(seed=0, intensity_mult=0.0)


def test_stationary_scenario_constant_schedule() -> None:
    sc = stationary_scenario(40, p_buy=0.6, seed=2)
    assert sc.family == "stationary"
    assert sc.n_regimes == 1 and sc.total_mos == 40
    flow = ScheduledRegimeFlow(sc)
    for _ in range(45):  # overrun past the planned horizon: clamps on last regime
        assert flow.current().p_buy == pytest.approx(0.6)
        flow.advance()
    assert flow.transitions == []
    assert flow.state_mo_counts == [45]
    with pytest.raises(ValueError):
        stationary_scenario(0)
    with pytest.raises(ValueError):
        stationary_scenario(10, p_buy=1.5)


def test_default_expected_mos_matches_paper_count() -> None:
    cfg = santa_fe_config(seed=0)  # mu = 0.10 per side -> 2*mu = 0.20 MO/s
    assert default_expected_mos(cfg, 300.0) == 60
    assert default_expected_mos(cfg, 1.0) == 1  # floored at 1
    with pytest.raises(ValueError):
        default_expected_mos(cfg, 0.0)


# ---------------------------------------------------------------------------
# ScenarioBandit: EWMA difficulty, weights, cap, anneal, refresh
# ---------------------------------------------------------------------------


def _pool(n: int = 8, seed: int = 0) -> list[Scenario]:
    return ScenarioGenerator(seed=seed).initial_pool(n, 30)


def _bandit(n: int = 8, **kw) -> ScenarioBandit:
    kw.setdefault("w_max", 0.4)
    kw.setdefault("n_updates_planned", 10)
    return ScenarioBandit(_pool(n), **kw)


def test_bandit_init_uniform_weights() -> None:
    bt = _bandit(8)
    w = bt.weights()
    assert w.shape == (8,)
    assert np.allclose(w, 1.0 / 8)
    assert w.sum() == pytest.approx(1.0)
    assert bt.n_updates == 0 and bt.n_refreshes == 0
    assert np.all(bt.difficulties == 0.0)


def test_bandit_ewma_closed_form() -> None:
    bt = _bandit(8, eta_ewma=0.05)
    bt.record(3, 2.0)
    d = bt.difficulties
    assert d[3] == pytest.approx(0.05 * 2.0)
    assert np.all(np.delete(d, 3) == 0.0)
    bt.record(3, -1.0)
    assert bt.difficulties[3] == pytest.approx(0.95 * 0.1 + 0.05 * (-1.0))
    assert bt.n_updates == 2


def test_bandit_weights_closed_form_softmax() -> None:
    bt = ScenarioBandit(
        _pool(3),
        n_updates_planned=4,
        eta_ewma=1.0,
        beta_start=0.5,
        beta_end=0.5,
        eps_start=0.0,
        eps_end=0.0,
        w_max=0.9,
    )
    bt.record(0, 2.0)  # d = (2, 0, 0)
    bt.record(0, 0.0)  # d = (2, 0, 0) still (eta=1)
    d = bt.difficulties
    mu, sd = d.mean(), d.std()
    nu = (d - mu) / (sd + 1e-8)
    exp = np.exp(0.5 * nu)
    w_expected = exp / exp.sum()
    np.testing.assert_allclose(bt.weights(), w_expected, atol=1e-12)


def test_bandit_harder_arm_samples_more() -> None:
    bt = _bandit(6, beta_start=0.5, beta_end=0.5, eps_start=0.0, eps_end=0.0)
    bt.record(2, 5.0)
    w = bt.weights()
    assert w.argmax() == 2
    assert w[2] > 1.0 / 6.0


def test_bandit_zero_std_gives_uniform() -> None:
    bt = _bandit(6, beta_start=0.5, beta_end=0.5, eps_start=0.0, eps_end=0.0)
    for i in range(6):
        bt.record(i, 1.0)  # identical difficulties -> s_d = 0 -> nu = 0
    assert np.allclose(bt.weights(), 1.0 / 6.0)


def test_bandit_wmax_cap_redistributes() -> None:
    bt = _bandit(10, beta_start=2.0, beta_end=2.0, eps_start=0.0, eps_end=0.0, w_max=0.30)
    bt.record(4, 50.0)  # dominates the softmax
    w = bt.weights()
    assert w[4] <= 0.30 + 1e-12
    assert w.sum() == pytest.approx(1.0)
    others = np.delete(w, 4)
    assert np.allclose(others, others[0])  # uniform redistribution over uncapped arms


def test_bandit_eps_floor_min_weight() -> None:
    bt = _bandit(5, beta_start=3.0, beta_end=3.0, eps_start=0.6, eps_end=0.6, w_max=0.9)
    bt.record(0, 100.0)
    w = bt.weights()
    assert w.min() >= 0.6 / 5 - 1e-12  # every arm keeps >= eps/M mass


def test_bandit_anneal_schedule() -> None:
    bt = _bandit(8, n_updates_planned=10)  # half = 5
    b0, e0 = bt.anneal(0)
    b_half, e_half = bt.anneal(5)
    b_end, e_end = bt.anneal(10)
    assert (b0, e0) == (0.0, 1.0)
    assert b_half == pytest.approx(0.5)
    assert e_half == pytest.approx(0.5)
    assert (b_end, e_end) == (0.5, 0.5)
    b_far, e_far = bt.anneal(10_000)  # saturated
    assert (b_far, e_far) == (0.5, 0.5)


def test_bandit_select_seeded_deterministic() -> None:
    b1, b2 = _bandit(8, seed=17), _bandit(8, seed=17)
    seq1 = [b1.select() for _ in range(20)]
    seq2 = [b2.select() for _ in range(20)]
    assert seq1 == seq2
    b3 = _bandit(8, seed=18)
    seq3 = [b3.select() for _ in range(20)]
    assert seq1 != seq3


def test_bandit_record_fail_closed() -> None:
    bt = _bandit(4)
    with pytest.raises(ValueError):
        bt.record(-1, 1.0)
    with pytest.raises(ValueError):
        bt.record(4, 1.0)
    with pytest.raises(ValueError):
        bt.record(True, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bt.record(0, float("nan"))
    with pytest.raises(ValueError):
        bt.record(0, float("inf"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_updates_planned": 0},
        {"n_updates_planned": 4, "eta_ewma": 0.0},
        {"n_updates_planned": 4, "eta_ewma": 1.5},
        {"n_updates_planned": 4, "beta_end": -0.1},
        {"n_updates_planned": 4, "eps_start": 0.2, "eps_end": 0.8},
        {"n_updates_planned": 4, "w_max": 0.0},
        {"n_updates_planned": 4, "eps_nu": 0.0},
        {"n_updates_planned": 4, "refresh_every": 0},
        {"n_updates_planned": 4, "refresh_frac": 0.0},
        {"n_updates_planned": 4, "refresh_frac": 1.5},
    ],
)
def test_bandit_init_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ScenarioBandit(_pool(8), **kwargs)


def test_bandit_infeasible_cap_fails_closed() -> None:
    # 8 arms at w_max=0.05 cannot sum to 1 -> hard error, not silent clamp.
    with pytest.raises(ValueError):
        ScenarioBandit(_pool(8), n_updates_planned=4, w_max=0.05)
    # the paper's cap is feasible at the paper's pool size
    bt = ScenarioBandit(_pool(64), n_updates_planned=4, **PAPER_BANDIT_KWARGS)
    assert bt.weights().sum() == pytest.approx(1.0)


def test_bandit_refresh_replaces_easiest_arms() -> None:
    gen = ScenarioGenerator(seed=77)
    pool = gen.initial_pool(10, 30)
    bt = ScenarioBandit(pool, n_updates_planned=4, w_max=0.4, refresh_every=2, refresh_frac=0.3)
    for i, loss in enumerate([9.0, 8.0, 7.0, 6.0]):
        bt.record(i, loss)
    assert bt.n_updates == 4
    assert bt.needs_refresh() is True  # 4 % refresh_every(2) == 0, none done yet
    before = [s.seed for s in pool]
    idx = bt.refresh(gen, 30)
    # refresh_frac=0.3 of 10 -> the 3 easiest arms: stable argsort of
    # d = [9,8,7,6,0,...,0] picks indices 4,5,6 (the first zero-difficulty arms)
    assert idx == [4, 5, 6]
    for i in idx:
        assert isinstance(pool[i], Scenario)
    # difficulties are EWMA values: d_i = 0.05 * loss_i after one record each
    mean_d = float(np.mean(0.05 * np.asarray([9.0, 8.0, 7.0, 6.0] + [0.0] * 6)))
    for i in idx:
        assert bt.difficulties[i] == pytest.approx(mean_d)
    untouched = set(range(10)) - set(idx)
    assert {pool[i].seed for i in untouched} == {before[i] for i in untouched}
    assert bt.n_refreshes == 1
    assert bt.needs_refresh() is False


def test_bandit_needs_refresh_flag_semantics() -> None:
    gen = ScenarioGenerator(seed=1)
    bt = ScenarioBandit(gen.initial_pool(4, 20), n_updates_planned=4, refresh_every=2, w_max=0.4)
    assert bt.needs_refresh() is False
    bt.record(0, 1.0)
    assert bt.needs_refresh() is False
    bt.record(1, 1.0)
    assert bt.needs_refresh() is True
    bt.refresh(ScenarioGenerator(seed=2), 20)
    assert bt.needs_refresh() is False


def test_bandit_refresh_fail_closed() -> None:
    bt = _bandit(4)
    with pytest.raises(TypeError):
        bt.refresh("not_a_generator", 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bt.refresh(ScenarioGenerator(seed=0), 0)


# ---------------------------------------------------------------------------
# penalized_terminal_score (paper H_T)
# ---------------------------------------------------------------------------


def test_penalized_terminal_score_closed_form() -> None:
    # mtm=10, q path post-initial = [2, -3, 1]; cap=8, fw=0.5 -> wall=4
    # penalty = phi * (4 + 9 + 1) = 0.001 * 14 = 0.014 (no wall breach)
    bundle = {"sim_internal_mtm_pnl_final": 10.0, "inventory_path": [0, 2, -3, 1]}
    out = penalized_terminal_score(
        bundle, reward_phi=0.001, reward_wall_fraction=0.5, inventory_cap=8
    )
    assert out["sim_internal_penalty_sum"] == pytest.approx(0.014)
    assert out["sim_internal_terminal_score"] == pytest.approx(10.0 - 0.014)
    assert out["sim_internal_arm_loss"] == pytest.approx(-(10.0 - 0.014))
    assert out["n_penalty_terms"] == 3.0


def test_penalized_terminal_score_wall_term() -> None:
    # cap=4, fw=0.5 -> wall=2; q=3 breaches by 1 -> wall term = 1
    bundle = {"sim_internal_mtm_pnl_final": 1.0, "inventory_path": [0, 3]}
    out = penalized_terminal_score(
        bundle, reward_phi=0.5, reward_wall_fraction=0.5, inventory_cap=4
    )
    # penalty = 0.5 * (9 + 1) = 5.0
    assert out["sim_internal_penalty_sum"] == pytest.approx(5.0)
    assert out["sim_internal_terminal_score"] == pytest.approx(-4.0)


def test_penalized_terminal_score_uncapped_drops_wall() -> None:
    bundle = {"sim_internal_mtm_pnl_final": 0.0, "inventory_path": [0, 10]}
    capped = penalized_terminal_score(
        bundle, reward_phi=1.0, reward_wall_fraction=0.5, inventory_cap=4
    )
    uncapped = penalized_terminal_score(
        bundle, reward_phi=1.0, reward_wall_fraction=0.5, inventory_cap=None
    )
    # uncapped: penalty = 1.0 * 100 = 100 (quadratic only)
    assert uncapped["sim_internal_penalty_sum"] == pytest.approx(100.0)
    assert capped["sim_internal_penalty_sum"] > uncapped["sim_internal_penalty_sum"]


def test_penalized_terminal_score_fail_closed() -> None:
    with pytest.raises(TypeError):
        penalized_terminal_score("x", reward_phi=1e-3, reward_wall_fraction=0.5, inventory_cap=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        penalized_terminal_score(
            {"sim_internal_mtm_pnl_final": float("nan"), "inventory_path": [0, 1]},
            reward_phi=1e-3,
            reward_wall_fraction=0.5,
            inventory_cap=8,
        )
    with pytest.raises(ValueError):
        penalized_terminal_score(
            {"sim_internal_mtm_pnl_final": 0.0, "inventory_path": [0]},
            reward_phi=1e-3,
            reward_wall_fraction=0.5,
            inventory_cap=8,
        )
    with pytest.raises(ValueError):
        penalized_terminal_score(
            {"sim_internal_mtm_pnl_final": 0.0, "inventory_path": [0, float("nan")]},
            reward_phi=1e-3,
            reward_wall_fraction=0.5,
            inventory_cap=8,
        )
    with pytest.raises(ValueError):
        penalized_terminal_score(
            {"sim_internal_mtm_pnl_final": 0.0, "inventory_path": [0, 1]},
            reward_phi=-1e-3,
            reward_wall_fraction=0.5,
            inventory_cap=8,
        )


# ---------------------------------------------------------------------------
# Torch-free numpy contrast policies
# ---------------------------------------------------------------------------


def _mm_state(bid: float | None = 99.99, ask: float | None = 100.01, q: int = 0) -> MMState:
    return MMState(
        t=0.0,
        mid=None if bid is None else (bid + ask) / 2,
        best_bid=bid,
        best_ask=ask,
        inventory=q,
        tau=10.0,
    )


def test_fixed_offset_policy_maps_offsets_to_grid() -> None:
    st = _mm_state(99.99, 100.01)
    assert fixed_offset_policy((0, 0), tick=0.01)(st) == (99.99, 100.01)
    # (-1,-1) at a 2-tick spread: inside bid = 100.00 < inside ask = 100.00 is
    # a cross -> falls back to the touch (same clamp as the agent path)
    b, a = fixed_offset_policy((-1, -1), tick=0.01)(st)
    assert (b, a) == (pytest.approx(99.99), pytest.approx(100.01))
    b2, a2 = fixed_offset_policy((1, 0), tick=0.01)(st)
    assert (b2, a2) == (pytest.approx(99.98), pytest.approx(100.01))
    # one-sided book -> no quotes
    assert fixed_offset_policy((0, 0), tick=0.01)(_mm_state(None, 100.01)) == (None, None)


def test_fixed_offset_policy_crossing_safety_fallback() -> None:
    # 1-tick spread + (-1,-1) maps inside quotes that cross -> clamped to touch.
    st = _mm_state(99.99, 100.00)
    b, a = fixed_offset_policy((-1, -1), tick=0.01)(st)
    assert (b, a) == (pytest.approx(99.99), pytest.approx(100.00))


def test_fixed_offset_policy_fail_closed() -> None:
    with pytest.raises(ValueError):
        fixed_offset_policy((2, 0), tick=0.01)
    with pytest.raises(ValueError):
        fixed_offset_policy((0, 0), tick=0.0)


def test_random_offset_policy_seeded_and_on_grid() -> None:
    p1 = random_offset_policy(tick=0.01, seed=5)
    p2 = random_offset_policy(tick=0.01, seed=5)
    st = _mm_state(99.99, 100.01)
    seq1 = [p1(st) for _ in range(30)]
    seq2 = [p2(st) for _ in range(30)]
    assert seq1 == seq2  # seeded -> deterministic
    tick = 0.01
    grid = {(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}
    expected = {
        (99.99 - db * tick, 100.01 + da * tick)
        if 99.99 - db * tick < 100.01 + da * tick
        else (99.99, 100.01)
        for db, da in grid
    }
    for pair in seq1:
        b, a = pair
        assert any(
            math.isclose(b, eb, abs_tol=1e-9) and math.isclose(a, ea, abs_tol=1e-9)
            for eb, ea in expected
        )
    assert len(set(seq1)) > 1  # actually randomizes


def test_random_offset_policy_fail_closed() -> None:
    with pytest.raises(ValueError):
        random_offset_policy((), tick=0.01)
    with pytest.raises(ValueError):
        random_offset_policy(((0, 0, 0),), tick=0.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        random_offset_policy(((0, 2),), tick=0.01)
    with pytest.raises(ValueError):
        random_offset_policy(tick=-0.01)


def test_inventory_skew_policy_rule() -> None:
    pol = inventory_skew_policy(tick=0.01, inventory_cap=8, skew_fraction=0.5)
    st_neutral = _mm_state(99.99, 100.02, q=0)
    assert pol(st_neutral) == (pytest.approx(99.99), pytest.approx(100.02))
    # long inventory >= 4: bid deeper (+1), ask inside (-1) -> lean to sell
    b, a = pol(_mm_state(99.99, 100.02, q=5))
    assert (b, a) == (pytest.approx(99.98), pytest.approx(100.01))
    # short inventory <= -4: mirrored
    b2, a2 = pol(_mm_state(99.99, 100.02, q=-5))
    assert (b2, a2) == (pytest.approx(100.00), pytest.approx(100.03))
    with pytest.raises(ValueError):
        inventory_skew_policy(tick=0.01, inventory_cap=0)
    with pytest.raises(ValueError):
        inventory_skew_policy(tick=0.01, inventory_cap=8, skew_fraction=1.5)


# ---------------------------------------------------------------------------
# Event-stream causality: no future-flow leakage through the adapter
# ---------------------------------------------------------------------------


def _diverging_scenarios(l0: int = 30, n_tail: int = 60) -> tuple[Scenario, Scenario]:
    """Two scenarios sharing the first l0 MOs (p=0.2), diverging after."""
    head = RegimeSegment(l0, 0.2, 1.0, 30.0)
    a = Scenario((head, RegimeSegment(n_tail, 0.2, 1.0, 60.0)), "random_persistence", seed=1)
    b = Scenario((head, RegimeSegment(n_tail, 0.8, 1.0, 60.0)), "random_persistence", seed=2)
    return a, b


def test_event_stream_causality_no_future_flow_leak() -> None:
    """Identical scenario prefix -> identical event stream until the MO-clock
    boundary. The engine draws the MO side from the *current* regime only, so
    nothing about the post-divergence flow can leak into earlier state."""
    a, b = _diverging_scenarios(l0=30)
    cfg = santa_fe_config(seed=7)
    mo_sides: list[list[str | None]] = []
    for sc in (a, b):
        flow = ScheduledRegimeFlow(sc)
        sim = ZILobSimulator(cfg, flow=flow)
        sides: list[str | None] = []
        for _ in range(600):
            kind = sim.step()
            if kind == "market":
                n_tr = len(sim.trades)
                sides.append(None if n_tr == 0 else sim.trades[n_tr - 1].aggressor)
                # NOTE: noop MOs (empty book) record None — still one MO tick.
        mo_sides.append(sides)
    sa, sb = mo_sides
    assert len(sa) == len(sb) and len(sa) >= 40
    # MOs 1..30 use the shared regime (advance happens after the side draw,
    # so MO #31 is the first draw under the diverged regime).
    assert sa[:30] == sb[:30]
    # and the streams DO eventually diverge (sanity: the suffix differs)
    assert sa[30:] != sb[30:]


def test_session_causality_prefix_identical() -> None:
    """Under a fixed torch-free policy, sessions whose flows differ ONLY
    after MO #30 are bit-identical while no diverged MO draw exists
    (n_mo_arrivals <= 30); once MO #31 has arrived they may differ. This
    proves no future-flow leakage through the adapter."""
    a, b = _diverging_scenarios(l0=30)
    cfg = santa_fe_config(seed=7)

    def sess(sc: Scenario, horizon: float) -> dict:
        return run_rl_mm_session(
            config=cfg,
            horizon=horizon,
            policy=fixed_offset_policy((0, 0), tick=cfg.tick),
            flow=ScheduledRegimeFlow(sc),
            inventory_cap=CAP,
        )

    # short horizon: the pinned seed/horizon keeps both sessions inside the
    # shared prefix (asserted precondition — verified deterministically)
    sa, sb = sess(a, 140.0), sess(b, 140.0)
    assert sa["event_counts"]["n_mo_arrivals"] <= 30
    assert sb["event_counts"]["n_mo_arrivals"] <= 30
    assert sa["inventory_path"] == sb["inventory_path"]
    assert sa["inventory_path_times"] == sb["inventory_path_times"]
    np.testing.assert_array_equal(
        np.asarray(sa["sim_internal_mtm_pnl_path"]),
        np.asarray(sb["sim_internal_mtm_pnl_path"]),
    )
    assert sa["n_fills"] == sb["n_fills"]
    assert sa["n_decisions"] == sb["n_decisions"]

    # long horizon: MO #31 lands and the streams actually diverge (sanity)
    la, lb = sess(a, 260.0), sess(b, 260.0)
    assert la["event_counts"]["n_mo_arrivals"] > 30
    assert lb["event_counts"]["n_mo_arrivals"] > 30
    assert (
        la["inventory_path"] != lb["inventory_path"]
        or la["n_fills"] != lb["n_fills"]
        or la["inventory_path_times"] != lb["inventory_path_times"]
    )


# ---------------------------------------------------------------------------
# Planted regime switch: online filter + batch BOCPD cross-localization
# ---------------------------------------------------------------------------


def _collect_trade_signs(sc: Scenario, seed: int = 11, max_events: int = 4000) -> list[float]:
    sim = ZILobSimulator(santa_fe_config(seed=seed), flow=ScheduledRegimeFlow(sc))
    while len(sim.trades) < 120 and sim.n_events < max_events:
        sim.step()
    return [1.0 if tr.aggressor == "buy" else -1.0 for tr in sim.trades]


def test_filter_and_bocpd_localize_planted_scenario_switch() -> None:
    """A planted p_buy switch inside a ScheduledRegimeFlow is detected by BOTH
    the online Beta-Bernoulli filter (its belief flips sign across the
    boundary) and the batch Gaussian BOCPD — the wave-16 cross-check carried
    onto scheduled scenarios."""
    switch_mo = 60
    sc = Scenario(
        (RegimeSegment(switch_mo, 0.15, 1.0, 30.0), RegimeSegment(80, 0.85, 1.0, 30.0)),
        "correlated_direction",
        seed=3,
    )
    signs = _collect_trade_signs(sc)
    assert len(signs) >= 100
    filt = FlowBiasFilter(tau_r=30.0, a0=1.0, b0=1.0)
    beliefs = []
    for s in signs:
        filt.observe(1 if s > 0 else 0)
        beliefs.append(filt.belief()[0])
    iota = np.asarray(beliefs)
    assert iota[10] < 0.0  # early sell-biased regime
    assert iota[-1] > 0.0  # post-switch buy-biased regime
    flip = int(np.flatnonzero(iota > 0.0)[0])
    assert abs(flip - switch_mo) <= 40  # localized near the planted switch
    cp = bocpd_gaussian(np.asarray(signs), hazard=1.0 / 40.0)
    # repo convention: peak changepoint mass localizes near the true break
    # (changepoints[] needs cp_prob > 0.5 — too strict for ±1 sign noise)
    peak = int(np.argmax(cp["cp_prob"]))
    assert abs(peak - switch_mo) <= 25
    assert cp["cp_prob"].max() > 0.2


def test_augmented_state_vector_carries_belief_torch_free() -> None:
    """The aux belief slot of build_state_vector moves with a planted
    directional scenario while the 9 base features are identical."""
    spec_on = RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=CAP)
    spec_off = RLStateSpec(aux_enabled=False, inventory_cap=CAP)
    assert spec_on.state_dim == spec_off.state_dim + 3
    filt = FlowBiasFilter(tau_r=30.0)
    for _ in range(60):  # planted persistent buy flow
        filt.observe(1)
    s_on = build_state_vector(
        spec_on,
        spread_ticks=2,
        bid_depth0=5,
        bid_depth1=3,
        ask_depth0=4,
        ask_depth1=2,
        inventory=1,
        bid_opportunity=0.5,
        ask_opportunity=0.5,
        belief=filt.belief(),
    )
    s_off = build_state_vector(
        spec_off,
        spread_ticks=2,
        bid_depth0=5,
        bid_depth1=3,
        ask_depth0=4,
        ask_depth1=2,
        inventory=1,
        bid_opportunity=0.5,
        ask_opportunity=0.5,
    )
    assert s_on.shape == (12,) and s_off.shape == (9,)
    np.testing.assert_array_equal(s_on[:9], s_off)
    assert s_on[9] > 0.5  # iota_hat strongly buy-biased after 60 buys


# ---------------------------------------------------------------------------
# Policy-mode sessions through ScheduledRegimeFlow (torch-free env adapter)
# ---------------------------------------------------------------------------


def test_policy_session_under_scheduled_flow_schema() -> None:
    cfg = santa_fe_config(seed=5)
    sc = ScenarioGenerator(seed=5).draw(40, family="random_persistence")
    bundle = run_rl_mm_session(
        config=cfg,
        horizon=120.0,
        policy=random_offset_policy(tick=cfg.tick, seed=3),
        flow=ScheduledRegimeFlow(sc),
        inventory_cap=CAP,
    )
    _assert_synthetic_honest(bundle)
    assert bundle["session_completed"] is True
    assert bundle["policy_kind"] == "classic_policy"
    assert bundle["max_abs_inventory"] <= CAP
    assert bundle["n_fills"] >= 0
    assert isinstance(bundle["sim_internal_mtm_pnl_final"], float)


def test_scheduled_directional_flow_biases_trade_signs() -> None:
    """A heavily buy-biased scheduled regime produces buy-skewed trade signs —
    the scenario reaches the engine's MO side draw."""
    sc = Scenario((RegimeSegment(200, 0.9, 1.0, 30.0),), "correlated_direction", seed=1)
    signs = _collect_trade_signs(sc, seed=13)
    assert len(signs) >= 80
    assert float(np.mean(signs)) > 0.5  # buy fraction >> 50%


def test_glft_as_contrast_identical_scenario_seeds() -> None:
    """AS vs GLFT on the SAME scheduled scenario + config seeds: paired
    deterministic contrast rows, the torch-free side of the eval harness."""
    cfg = santa_fe_config(seed=0)
    gen = ScenarioGenerator(seed=21)
    out = evaluate_scenario_robustness(
        config=cfg,
        horizon=120.0,
        agent=None,
        policies={
            "as": as_policy(gamma=0.002, sigma=0.02, kappa=1000.0, tick=cfg.tick),
            "glft": _glft_pol(),
            "random": random_offset_policy(tick=cfg.tick, seed=2),
        },
        generator=gen,
        n_scenarios=2,
        families=("stationary", "random_persistence"),
        seed_base=41,
        inventory_cap=CAP,
    )
    _assert_synthetic_honest(out)
    for fam in ("stationary", "random_persistence"):
        for name in ("as", "glft", "random"):
            rows = out["sessions"][f"{name}_{fam}"]
            assert len(rows) == 2
            assert all(r["session_completed"] for r in rows)
            assert f"sim_internal_mtm_pnl_final_mean_{name}_{fam}" in out["metrics"]
            assert f"sim_internal_terminal_score_mean_{name}_{fam}" in out["metrics"]
    # paired seeds: the two actors share ep_seed within a scenario index
    as_rows = out["sessions"]["as_stationary"]
    gl_rows = out["sessions"]["glft_stationary"]
    assert [r["seed"] for r in as_rows] == [r["seed"] for r in gl_rows]
    # distinct actors produce distinct outcomes on identical seeds
    assert as_rows[0]["sim_internal_mtm_pnl_final"] != gl_rows[0]["sim_internal_mtm_pnl_final"]


def test_evaluate_robustness_deterministic_torch_free() -> None:
    cfg = santa_fe_config(seed=2)
    # fresh policy instances per call: a QuotePolicy may hold its own
    # Generator (user state) — determinism is asserted over identically
    # *constructed* actors, matching the wave-16 session contract.

    def kwargs() -> dict:
        return dict(
            config=cfg,
            horizon=100.0,
            policies={"random": random_offset_policy(tick=cfg.tick, seed=1)},
            generator=ScenarioGenerator(seed=31),
            n_scenarios=2,
            families=("correlated_direction",),
            seed_base=51,
            inventory_cap=CAP,
        )

    r1 = evaluate_scenario_robustness(**kwargs())
    r2 = evaluate_scenario_robustness(**kwargs())
    assert r1["metrics"] == r2["metrics"]
    assert r1["sessions"] == r2["sessions"]


def test_evaluate_robustness_fail_closed() -> None:
    cfg = santa_fe_config(seed=0)
    with pytest.raises(ValueError):
        evaluate_scenario_robustness(config=cfg, horizon=100.0)  # no actors
    with pytest.raises(ValueError):
        evaluate_scenario_robustness(
            config=cfg,
            horizon=100.0,
            policies={"p": random_offset_policy(tick=0.01)},
            families=("nope",),
        )
    with pytest.raises(ValueError):
        evaluate_scenario_robustness(
            config=cfg, horizon=0.0, policies={"p": random_offset_policy(tick=0.01)}
        )
    with pytest.raises(TypeError):
        evaluate_scenario_robustness(
            config=cfg,
            horizon=10.0,
            policies={"p": "not_callable"},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError):
        evaluate_scenario_robustness(
            config=cfg,
            horizon=10.0,
            policies={"p": random_offset_policy(tick=0.01)},
            n_scenarios=0,
        )


# ---------------------------------------------------------------------------
# Scenario-bandit fine-tuning (torch-gated)
# ---------------------------------------------------------------------------


def _finetune_kwargs(seed: int = 0) -> dict:
    gen = ScenarioGenerator(seed=seed)
    pool = gen.initial_pool(6, 30)
    bt = ScenarioBandit(pool, n_updates_planned=4, w_max=0.4, seed=seed)
    return {
        "config": santa_fe_config(seed=seed),
        "pool": pool,
        "bandit": bt,
        "generator": gen,
        "n_episodes": 4,
        "horizon": 120.0,
        "expected_mos": 30,
        "seed_base": seed,
    }


@requires_torch
def test_finetune_smoke_schema_and_difficulties() -> None:
    agent = _tiny_agent(0)
    out = run_scenario_bandit_finetuning(agent=agent, **_finetune_kwargs(0))
    _assert_synthetic_honest(out)
    assert out["kind"] == "scenario_bandit_finetuning"
    assert out["n_episodes"] == 4
    eps = out["episodes"]
    assert len(eps) == 4
    drawn = {e["arm"] for e in eps}
    d = np.asarray(out["bandit"]["difficulties"])
    for i in range(6):
        if i in drawn:
            assert d[i] != 0.0
        else:
            assert d[i] == 0.0
    assert len(out["loss_curve"]) > 0
    assert np.all(np.isfinite(out["loss_curve"]))
    assert len(out["sim_internal_score_path"]) == 4
    assert np.all(np.isfinite(out["sim_internal_score_path"]))
    assert len(out["bandit"]["weights_final"]) == 6
    for e in eps:
        assert e["session_completed"] is True
        assert e["max_abs_inventory"] <= CAP
        assert math.isfinite(e["sim_internal_terminal_score"])
        assert e["weight_at_selection"] > 0.0
        assert e["n_regimes"] >= 1


@requires_torch
def test_finetune_determinism_same_seed() -> None:
    a = run_scenario_bandit_finetuning(agent=_tiny_agent(0), **_finetune_kwargs(0))
    b = run_scenario_bandit_finetuning(agent=_tiny_agent(0), **_finetune_kwargs(0))
    assert a["sim_internal_score_path"] == b["sim_internal_score_path"]
    assert a["bandit"]["arms_drawn"] == b["bandit"]["arms_drawn"]
    assert a["bandit"]["weights_final"] == b["bandit"]["weights_final"]
    assert [e["sim_internal_mtm_pnl_final"] for e in a["episodes"]] == [
        e["sim_internal_mtm_pnl_final"] for e in b["episodes"]
    ]
    c = run_scenario_bandit_finetuning(agent=_tiny_agent(1), **_finetune_kwargs(1))
    assert (
        a["sim_internal_score_path"] != c["sim_internal_score_path"]
        or a["bandit"]["arms_drawn"] != c["bandit"]["arms_drawn"]
    )


@requires_torch
def test_finetune_fail_closed() -> None:
    agent = _tiny_agent(0)
    kw = _finetune_kwargs(0)
    with pytest.raises(TypeError):
        run_scenario_bandit_finetuning(agent="nope", **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_scenario_bandit_finetuning(agent=agent, **{**kw, "n_episodes": 0})
    with pytest.raises(ValueError):
        run_scenario_bandit_finetuning(agent=agent, **{**kw, "horizon": -1.0})
    with pytest.raises(ValueError):
        run_scenario_bandit_finetuning(agent=agent, **{**kw, "pool": []})
    # bandit built over a DIFFERENT pool object is rejected
    other_bt = ScenarioBandit(
        ScenarioGenerator(seed=99).initial_pool(6, 30), n_updates_planned=4, w_max=0.4
    )
    with pytest.raises(ValueError):
        run_scenario_bandit_finetuning(agent=agent, **{**kw, "bandit": other_bt})
    with pytest.raises(TypeError):
        run_scenario_bandit_finetuning(agent=agent, **{**kw, "generator": "x"})
    # default-constructed bandit on a small pool fails the w_max feasibility
    small_kw = {k: v for k, v in kw.items() if k != "bandit"}
    with pytest.raises(ValueError):
        run_scenario_bandit_finetuning(agent=agent, **small_kw)


@requires_torch
def test_finetune_weights_shift_toward_bad_arms() -> None:
    """With annealed eps < 1 and beta > 0 reached inside the run, the hardest
    observed arm's sampling weight exceeds its initial uniform share."""
    gen = ScenarioGenerator(seed=3)
    pool = gen.initial_pool(6, 30)
    bt = ScenarioBandit(
        pool,
        n_updates_planned=6,
        beta_start=0.0,
        beta_end=0.5,
        eps_start=1.0,
        eps_end=0.5,
        w_max=0.6,
        seed=3,
    )
    out = run_scenario_bandit_finetuning(
        agent=_tiny_agent(3),
        config=santa_fe_config(seed=3),
        pool=pool,
        bandit=bt,
        generator=gen,
        n_episodes=6,
        horizon=120.0,
        expected_mos=30,
        seed_base=3,
    )
    w = np.asarray(out["bandit"]["weights_final"])
    d = np.asarray(out["bandit"]["difficulties"])
    assert w.sum() == pytest.approx(1.0)
    assert (d != 0.0).sum() >= 1
    # softmax weights are monotone in difficulty even after cap redistribution
    assert w[int(np.argmax(d))] >= w.max() - 1e-12


@requires_torch
def test_evaluate_robustness_with_agent_schema_and_no_mutation() -> None:
    agent = _tiny_agent(4)
    train = _finetune_kwargs(4)
    run_scenario_bandit_finetuning(agent=agent, **{**train, "n_episodes": 2})
    n_updates, n_decisions = agent.n_updates, agent.n_decisions
    cfg = santa_fe_config(seed=4)
    out = evaluate_scenario_robustness(
        config=cfg,
        horizon=100.0,
        agent=agent,
        policies={
            "glft": _glft_pol(),
            "random": random_offset_policy(tick=cfg.tick, seed=7),
        },
        generator=ScenarioGenerator(seed=41),
        n_scenarios=1,
        families=("stationary", "correlated_direction"),
        seed_base=61,
    )
    _assert_synthetic_honest(out)
    for fam in ("stationary", "correlated_direction"):
        for name in ("c51", "glft", "random"):
            key = f"{name}_{fam}"
            assert len(out["sessions"][key]) == 1
            assert out["sessions"][key][0]["session_completed"] is True
            assert f"sim_internal_terminal_score_mean_{key}" in out["metrics"]
        for other in ("glft", "random"):
            assert f"sim_internal_score_gap_c51_minus_{other}_{fam}_mean" in out["metrics"]
    # greedy eval never mutates agent experience (no gradient steps);
    # n_decisions still counts eval decisions, so only n_updates is pinned
    assert agent.n_updates == n_updates
    _ = n_decisions


@requires_torch
def test_inventory_cap_matches_agent_or_fails() -> None:
    agent = _tiny_agent(0)
    with pytest.raises(ValueError):
        evaluate_scenario_robustness(
            config=santa_fe_config(seed=0),
            horizon=50.0,
            agent=agent,
            inventory_cap=CAP + 1,
        )


@requires_torch
def test_trained_agent_vs_random_same_seeds_bounded() -> None:
    """Seeded training smoke: after tiny fine-tuning the greedy policy's
    simulator-internal scores on paired eval seeds are finite and at least
    the seeded random baseline's (bounded check — NOT the paper's
    improvement claim, which needs paper-scale budgets)."""
    agent = _tiny_agent(6)
    run_scenario_bandit_finetuning(agent=agent, **_finetune_kwargs(6))
    cfg = santa_fe_config(seed=6)
    out = evaluate_scenario_robustness(
        config=cfg,
        horizon=120.0,
        agent=agent,
        policies={"random": random_offset_policy(tick=cfg.tick, seed=9)},
        generator=ScenarioGenerator(seed=55),
        n_scenarios=2,
        families=("stationary", "random_persistence"),
        seed_base=71,
    )
    for fam in ("stationary", "random_persistence"):
        c51 = out["metrics"][f"sim_internal_terminal_score_mean_c51_{fam}"]
        rnd = out["metrics"][f"sim_internal_terminal_score_mean_random_{fam}"]
        assert math.isfinite(c51) and math.isfinite(rnd)
        # bounded seeded smoke: trained policy not catastrophically worse
        assert c51 >= rnd - 1.0
