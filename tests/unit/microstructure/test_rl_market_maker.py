"""Tests for microstructure/rl_market_maker.py — lane B4-ii.

Distributional-RL (C51) market maker over the lane B4-i ZI-LOB simulator,
following Moret & Lillo (2026, arXiv:2609.11614, verified against the arXiv
abstract + HTML full text): six-action quote-offset grid, SMDP reward with
inventory wall, Beta-Bernoulli Bayesian online flow-bias filter (Appendix A),
queue-adjusted quote-exposure imbalance (Eqs. 12-13), categorical projection
(Bellemare et al. 2017), epsilon-greedy + uniform replay (documented
deviations from the paper's NoisyNet/PER).

Everything is **labeled SYNTHETIC** correctness validation at TINY training
budgets — never market evidence, no live-trading claim. The tiny-budget tests
assert the plumbing and that a trained policy completes sessions with bounded
inventory; they do NOT assert that the trained policy beats GLFT (that
comparison lives in the ``slow``-marked benchmark test / ``rl_mm_benchmark``,
documented-optional for the bench battery). Simulator-internal accounting is
namespaced ``sim_internal_*`` and asserted here to never leak as a headline
metric. Torch tests skip cleanly when the ``nn`` extra is absent; the numpy
core (filter, buffer, state encoding, action grid, classic-policy sessions)
runs without torch.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace

import numpy as np
import pytest

from quant_fund.microstructure.rl_market_maker import (
    FULL_ACTION_GRID,
    PAPER_ACTION_GRID,
    C51Config,
    C51MarketMaker,
    FlowBiasFilter,
    ReplayBuffer,
    RLStateSpec,
    build_state_vector,
    evaluate_rl_market_makers,
    paper_regime_flow,
    quote_exposure_imbalance,
    rl_mm_benchmark,
    run_rl_mm_session,
    train_c51_market_maker,
    validate_action_grid,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
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
    not _HAS_TORCH, reason="C51 RL market maker training requires the nn extra (torch)"
)

# Forbidden *headline* metric tokens (mirrors research.catalog registry). "pnl"
# is permitted ONLY under the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

# ---------------------------------------------------------------------------
# Tiny-budget pins (tests must stay well under the 120 s lane budget)
# ---------------------------------------------------------------------------

CAP = 12
TINY_SPEC = RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=CAP)
TINY_C51 = C51Config(
    n_atoms=21,
    v_min=-3.0,
    v_max=3.0,
    hidden=(32,),
    gamma_event=0.999,
    n_step=3,
    lr=1e-3,
    batch_size=32,
    buffer_capacity=4096,
    target_update_every=25,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_steps=250,
    seed=0,
)
TRAIN_HORIZON = 600.0
TRAIN_EPISODES = 3
EVAL_HORIZON = 400.0


def _tiny_agent(seed: int = 0) -> C51MarketMaker:
    return C51MarketMaker(TINY_SPEC, replace(TINY_C51, seed=seed))


def _as_pol():
    return as_policy(gamma=0.002, sigma=0.02, kappa=1000.0, tick=0.01)


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


def _random_state(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.standard_normal(dim)


# ---------------------------------------------------------------------------
# FlowBiasFilter (paper Appendix A) — numpy core, no torch
# ---------------------------------------------------------------------------


def test_flow_filter_fail_closed() -> None:
    with pytest.raises(ValueError):
        FlowBiasFilter(tau_r=0.0)
    with pytest.raises(ValueError):
        FlowBiasFilter(tau_r=float("nan"))
    with pytest.raises(ValueError):
        FlowBiasFilter(a0=0.0)
    with pytest.raises(ValueError):
        FlowBiasFilter(b0=-1.0)
    with pytest.raises(ValueError):
        FlowBiasFilter(max_run=0)
    with pytest.raises(ValueError):
        FlowBiasFilter(hazard=0.0)
    with pytest.raises(ValueError):
        FlowBiasFilter(hazard=1.0)
    f = FlowBiasFilter(tau_r=10.0)
    with pytest.raises(ValueError):
        f.observe(2)
    with pytest.raises(ValueError):
        f.observe(-1)
    with pytest.raises(ValueError):
        f.observe(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        f.observe(True)  # bool rejected even though True == 1


def test_flow_filter_hazard_and_prior_belief() -> None:
    f = FlowBiasFilter(tau_r=60.0)
    # h = 1 - exp(-1/tau_r) from Exp(1/tau_r) segment durations (paper Eq. 25).
    assert f.hazard == pytest.approx(1.0 - math.exp(-1.0 / 60.0))
    assert f.max_run >= 64
    # Prior belief with a uniform Beta(1,1): no bias, zero run length.
    iota, ell = f.belief()
    assert iota == pytest.approx(0.0)
    assert ell == pytest.approx(0.0)
    assert f.changepoint_mass() == 0.0


def test_flow_filter_tracks_bias_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    balanced = rng.permutation(np.concatenate([np.ones(150), np.zeros(150)])).astype(int)
    f = FlowBiasFilter(tau_r=30.0)
    for y in balanced:
        f.observe(int(y))
    iota, _ = f.belief()
    assert abs(iota) < 0.15  # balanced flow: posterior bias near zero

    buys = np.ones(100, dtype=int)
    fa = FlowBiasFilter(tau_r=30.0)
    fb = FlowBiasFilter(tau_r=30.0)
    iotas = []
    for y in buys:
        fa.observe(int(y))
        fb.observe(int(y))
        iotas.append(fa.belief()[0])
    assert fa.belief() == fb.belief()  # determinism
    assert iotas[-1] > 0.5  # all-buy flow: strong positive bias
    assert iotas[-1] > iotas[9]  # belief accumulates evidence monotonically here
    _, ell = fa.belief()
    assert ell > 0.0


def test_flow_filter_resets_run_length_at_changepoint() -> None:
    f = FlowBiasFilter(tau_r=20.0)
    stream = [1] * 60 + [0] * 60
    beliefs = []
    for y in stream:
        f.observe(y)
        beliefs.append(f.belief())
    iota_before, ell_before = beliefs[58]  # deep inside the buy segment
    assert iota_before > 0.5
    assert ell_before > 10.0
    iota_after = beliefs[119][0]
    assert iota_after < -0.5  # tracks the direction flip
    ell_window = [b[1] for b in beliefs[60:75]]
    assert min(ell_window) < 8.0  # run-length posterior collapses at the change
    assert min(ell_window) < ell_before
    assert f.n_mo == 120
    assert f.changepoint_mass() >= 0.0


def test_flow_filter_localizes_change_like_bocpd_gaussian() -> None:
    """Composition check: models.changepoint.bocpd_gaussian (batch Gaussian
    Adams-MacKay BOCPD) and the online Beta-Bernoulli filter localize the
    same changepoint on a sticky signed flow."""
    signs = np.concatenate([np.ones(60), -np.ones(60)])
    hazard = 1.0 - math.exp(-1.0 / 20.0)
    out = bocpd_gaussian(signs, hazard=hazard, mu0=0.0)
    cps = out["changepoints"]
    assert len(cps) >= 1
    assert any(50 <= int(c) <= 75 for c in cps)
    f = FlowBiasFilter(tau_r=20.0)
    flip_by = None
    for i, s in enumerate(signs):
        f.observe(1 if s > 0 else 0)
        if i >= 60 and f.belief()[0] < -0.5:
            flip_by = i
            break
    assert flip_by is not None and 60 <= flip_by <= 85


# ---------------------------------------------------------------------------
# Quote-exposure imbalance (paper Eqs. 12-13) — numpy core
# ---------------------------------------------------------------------------


def test_quote_exposure_imbalance_values() -> None:
    both_front = quote_exposure_imbalance(
        bid_alive=True, bid_queue_ahead=0, ask_alive=True, ask_queue_ahead=0
    )
    assert both_front == pytest.approx(0.0)
    bid_only = quote_exposure_imbalance(
        bid_alive=True, bid_queue_ahead=0, ask_alive=False, ask_queue_ahead=None
    )
    assert bid_only == pytest.approx(0.5)  # paper range endpoint
    ask_only = quote_exposure_imbalance(
        bid_alive=False, bid_queue_ahead=None, ask_alive=True, ask_queue_ahead=0
    )
    assert ask_only == pytest.approx(-0.5)
    neither = quote_exposure_imbalance(
        bid_alive=False, bid_queue_ahead=None, ask_alive=False, ask_queue_ahead=None
    )
    assert neither == pytest.approx(0.0)
    buried_ask = quote_exposure_imbalance(
        bid_alive=True, bid_queue_ahead=0, ask_alive=True, ask_queue_ahead=3
    )
    # O_b = 1, O_a = 0.25 -> (1 - 0.25) / (1 + 1.25)
    assert buried_ask == pytest.approx(0.75 / 2.25)


def test_quote_exposure_imbalance_fail_closed() -> None:
    with pytest.raises(ValueError):
        quote_exposure_imbalance(
            bid_alive=True, bid_queue_ahead=None, ask_alive=False, ask_queue_ahead=None
        )
    with pytest.raises(ValueError):
        quote_exposure_imbalance(
            bid_alive=False, bid_queue_ahead=2, ask_alive=False, ask_queue_ahead=None
        )
    with pytest.raises(ValueError):
        quote_exposure_imbalance(
            bid_alive=True, bid_queue_ahead=-1, ask_alive=False, ask_queue_ahead=None
        )


# ---------------------------------------------------------------------------
# Action grid (paper Eq. 15) — numpy core
# ---------------------------------------------------------------------------


def test_action_grids_match_paper() -> None:
    assert PAPER_ACTION_GRID == ((-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
    assert len(FULL_ACTION_GRID) == 9
    assert (0, 0) in FULL_ACTION_GRID
    assert set(PAPER_ACTION_GRID) <= set(FULL_ACTION_GRID)
    assert validate_action_grid(PAPER_ACTION_GRID) == PAPER_ACTION_GRID
    assert validate_action_grid(FULL_ACTION_GRID) == FULL_ACTION_GRID


def test_validate_action_grid_fail_closed() -> None:
    with pytest.raises(ValueError):
        validate_action_grid(())
    with pytest.raises(ValueError):
        validate_action_grid(((0, 0), (2, 0)))  # offset out of range
    with pytest.raises(ValueError):
        validate_action_grid(((0, 0), (0, 0)))  # duplicate
    with pytest.raises(ValueError):
        validate_action_grid(((0, 0, 0),))  # not a pair
    with pytest.raises(ValueError):
        validate_action_grid(((0, 0.5),))  # non-int


# ---------------------------------------------------------------------------
# State spec + encoding — numpy core
# ---------------------------------------------------------------------------


def test_state_spec_dims_and_fail_closed() -> None:
    assert RLStateSpec(aux_enabled=True).state_dim == 12
    assert RLStateSpec(aux_enabled=False).state_dim == 9
    with pytest.raises(ValueError):
        RLStateSpec(inventory_cap=0)
    with pytest.raises(ValueError):
        RLStateSpec(filter_tau_r=0.0)
    with pytest.raises(ValueError):
        RLStateSpec(filter_a0=0.0)
    with pytest.raises(ValueError):
        RLStateSpec(filter_max_run=0)
    with pytest.raises(ValueError):
        RLStateSpec(depth_scale=0.0)
    with pytest.raises(ValueError):
        RLStateSpec(spread_scale=float("nan"))


def test_build_state_vector_values() -> None:
    spec = RLStateSpec(aux_enabled=True, filter_tau_r=60.0, inventory_cap=10)
    v = build_state_vector(
        spec,
        spread_ticks=4,
        bid_depth0=5,
        bid_depth1=0,
        ask_depth0=5,
        ask_depth1=2,
        inventory=10,
        bid_opportunity=1.0,
        ask_opportunity=0.25,
        belief=(0.5, 30.0),
    )
    assert v.shape == (12,)
    assert v[0] == pytest.approx(1.0)  # spread 4 / spread_scale 4
    assert v[1] == pytest.approx(0.5)  # depth 5 / depth_scale 10
    assert v[5] == pytest.approx(0.0)  # balanced touch imbalance
    assert v[6] == pytest.approx(1.0)  # inventory at cap
    assert v[9] == pytest.approx(0.5)  # iota_hat passthrough
    assert v[10] == pytest.approx(0.1)  # ell_bar/tau_r = 0.5, /5 clip scale
    assert v[11] == pytest.approx(0.75 / 2.25)  # exposure imbalance from O_b/O_a
    # Clipping keeps degenerate books bounded.
    v_clip = build_state_vector(
        spec,
        spread_ticks=1000,
        bid_depth0=10_000,
        bid_depth1=0,
        ask_depth0=0,
        ask_depth1=0,
        inventory=-10,
        bid_opportunity=0.0,
        ask_opportunity=1.0,
        belief=(-4.0, 1e6),
    )
    assert v_clip[0] == pytest.approx(8.0)
    assert v_clip[1] == pytest.approx(8.0)
    assert v_clip[6] == pytest.approx(-1.0)
    assert v_clip[9] == pytest.approx(-1.0)  # iota clipped to [-1, 1]
    assert v_clip[10] == pytest.approx(1.0)  # run-length feature clipped to 1
    assert np.all(np.isfinite(v_clip))


def test_build_state_vector_fail_closed() -> None:
    spec = RLStateSpec(aux_enabled=True, inventory_cap=10)
    kwargs = dict(
        spread_ticks=1,
        bid_depth0=1,
        bid_depth1=0,
        ask_depth0=1,
        ask_depth1=0,
        inventory=0,
        bid_opportunity=1.0,
        ask_opportunity=1.0,
        belief=(0.0, 1.0),
    )
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "spread_ticks": 0})
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "bid_depth0": -1})
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "inventory": 11})  # |q| > cap
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "bid_opportunity": 1.5})
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "belief": None})  # aux requires belief
    with pytest.raises(ValueError):
        build_state_vector(spec, **{**kwargs, "belief": (float("nan"), 1.0)})
    spec_off = RLStateSpec(aux_enabled=False, inventory_cap=10)
    with pytest.raises(ValueError):
        build_state_vector(spec_off, **kwargs)  # belief given but aux disabled
    v = build_state_vector(spec_off, **{**kwargs, "belief": None})
    assert v.shape == (9,)
    with pytest.raises(TypeError):
        build_state_vector("not-a-spec", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReplayBuffer — numpy core
# ---------------------------------------------------------------------------


def test_replay_buffer_shapes_wrap_and_determinism() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(4, 3, seed=7)
    assert len(buf) == 0
    for i in range(6):  # capacity 4 -> ring wraparound
        s = _random_state(rng, 3)
        buf.push(s, i % 2, float(i) * 0.01, s * 0.5, 0.99, i == 5)
    assert len(buf) == 4
    assert buf.capacity == 4
    batch = buf.sample(3)
    assert batch["states"].shape == (3, 3)
    assert batch["next_states"].shape == (3, 3)
    assert batch["actions"].shape == (3,)
    assert batch["rewards"].shape == (3,)
    assert batch["gammas"].shape == (3,)
    assert batch["dones"].shape == (3,)

    # Same seed + same pushes -> identical samples (bit-deterministic).
    def _filled(seed: int) -> ReplayBuffer:
        r = np.random.default_rng(1)
        b = ReplayBuffer(8, 3, seed=seed)
        for i in range(8):
            s = _random_state(r, 3)
            b.push(s, i % 3, float(i), s, 0.9, False)
        return b

    b1, b2, b3 = _filled(5), _filled(5), _filled(6)
    s1, s2, s3 = b1.sample(4), b2.sample(4), b3.sample(4)
    assert np.array_equal(s1["states"], s2["states"])
    assert np.array_equal(s1["actions"], s2["actions"])
    assert not np.array_equal(s1["states"], s3["states"])


def test_replay_buffer_fail_closed() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(0, 3)
    with pytest.raises(ValueError):
        ReplayBuffer(4, 0)
    buf = ReplayBuffer(4, 3, seed=0)
    good = np.zeros(3)
    with pytest.raises(ValueError):
        buf.push(np.zeros(2), 0, 0.0, np.zeros(2), 0.9, False)  # wrong dim
    with pytest.raises(ValueError):
        buf.push(np.full(3, np.nan), 0, 0.0, good, 0.9, False)  # non-finite
    with pytest.raises(ValueError):
        buf.push(good, -1, 0.0, good, 0.9, False)  # negative action
    with pytest.raises(ValueError):
        buf.push(good, 0, float("inf"), good, 0.9, False)  # non-finite reward
    with pytest.raises(ValueError):
        buf.push(good, 0, 0.0, good, 0.0, False)  # gamma must be > 0
    with pytest.raises(ValueError):
        buf.push(good, 0, 0.0, good, 1.5, False)  # gamma must be <= 1
    buf.push(good, 0, 0.0, good, 0.9, False)
    with pytest.raises(ValueError):
        buf.sample(2)  # batch > len
    with pytest.raises(ValueError):
        buf.sample(0)


# ---------------------------------------------------------------------------
# C51Config / RLStateSpec validation — pure python, no torch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_atoms": 1},
        {"v_min": 3.0, "v_max": 3.0},
        {"v_min": 1.0, "v_max": -1.0},
        {"v_min": float("nan")},
        {"hidden": ()},
        {"hidden": (0,)},
        {"hidden": (-4,)},
        {"gamma_event": 0.0},
        {"gamma_event": 1.0},
        {"gamma_event": 1.5},
        {"n_step": 0},
        {"lr": 0.0},
        {"lr": float("nan")},
        {"batch_size": 0},
        {"buffer_capacity": 0},
        {"batch_size": 200, "buffer_capacity": 100},
        {"target_update_every": 0},
        {"eps_start": 1.5},
        {"eps_end": -0.1},
        {"eps_start": 0.1, "eps_end": 0.5},
        {"eps_decay_steps": 0},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_c51_config_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        C51Config(**kwargs)


def test_paper_regime_flow_construction() -> None:
    flow = paper_regime_flow(seed=3, tau_r=60.0, omega=0.30)
    assert isinstance(flow, MarkovRegimeFlow)
    st = flow.current()
    assert st.name == "buy_pressure"
    assert st.p_buy == pytest.approx(0.8)
    with pytest.raises(ValueError):
        paper_regime_flow(seed=0, tau_r=0.0)
    with pytest.raises(ValueError):
        paper_regime_flow(seed=0, omega=0.6)  # p_buy would exceed 1
    with pytest.raises(ValueError):
        paper_regime_flow(seed=0, omega=-0.1)
    with pytest.raises(ValueError):
        paper_regime_flow(seed=-1)


# ---------------------------------------------------------------------------
# Session runner fail-closed edges (classic-policy path needs no torch)
# ---------------------------------------------------------------------------


def test_session_fail_closed_no_torch_edges() -> None:
    cfg = santa_fe_config(seed=1)
    pol = _glft_pol()
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0)  # neither agent nor policy
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, agent=None, policy=None)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, training=True)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=0.0, policy=pol)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, decision_interval=0.0)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, learn_every=0)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, reward_phi=-1.0)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, reward_wall_fraction=0.0)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, inventory_cap=0)
    with pytest.raises(TypeError):
        run_rl_mm_session(config="not-a-config", horizon=100.0, policy=pol)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy="not-callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_rl_mm_session(config=cfg, horizon=100.0, policy=pol, flow="not-a-flow")  # type: ignore[arg-type]


def test_policy_mode_session_schema_and_honesty() -> None:
    """Classic QuotePolicy through the RL runner: same accounting, RL block off.

    Runs without torch — the runner's baseline path is numpy-only.
    """
    out = run_rl_mm_session(
        config=santa_fe_config(seed=3),
        horizon=250.0,
        policy=_glft_pol(),
        decision_interval=1.0,
        inventory_cap=CAP,
    )
    assert out["label"] == "SYNTHETIC"
    assert out["data_source"] == "SYNTHETIC_ZI_LOB_v1"
    assert out["agent_revision"] == "SYNTHETIC_C51_MM_v1"
    assert out["research_only"] is True
    assert out["live_pnl_claim"] is False
    assert out["claim"] == "simulator_internal_diagnostic_only"
    assert out["policy_kind"] == "classic_policy"
    assert out["training"] is False
    assert out["session_completed"] is True
    assert out["max_abs_inventory"] <= CAP
    assert math.isfinite(out["sim_internal_mtm_pnl_final"])
    assert out["action_grid"] is None
    assert out["action_histogram"] is None
    assert out["sim_internal_reward_path"] is None
    assert out["loss_curve"] == []
    assert out["epsilon_initial"] is None
    assert out["filter_belief_final"] is None
    assert out["n_decisions"] >= 100
    assert len(out["inventory_path"]) == len(out["inventory_path_times"])
    # Smart quoting: unchanged targets keep resting orders, so cancels are
    # bounded by twice the number of decisions (both legs re-posted).
    assert out["n_mm_cancels"] <= 2 * out["n_decisions"]
    # AS baseline through the same runner also completes.
    out_as = run_rl_mm_session(
        config=santa_fe_config(seed=4),
        horizon=250.0,
        policy=_as_pol(),
        decision_interval=1.0,
        inventory_cap=CAP,
    )
    assert out_as["session_completed"] is True
    assert out_as["max_abs_inventory"] <= CAP


# ---------------------------------------------------------------------------
# C51MarketMaker (torch-gated)
# ---------------------------------------------------------------------------


@requires_torch
def test_agent_construction_shapes_and_atoms() -> None:
    agent = _tiny_agent(seed=0)
    assert agent.n_actions == 6
    assert agent.state_dim == 12
    assert agent.action_grid == PAPER_ACTION_GRID
    atoms = agent.atoms
    assert atoms.shape == (21,)
    assert atoms[0] == pytest.approx(-3.0)
    assert atoms[-1] == pytest.approx(3.0)
    assert np.all(np.diff(atoms) > 0)
    assert agent.epsilon == pytest.approx(1.0)
    assert agent.n_updates == 0
    assert agent.total_stored == 0
    with pytest.raises(TypeError):
        C51MarketMaker("not-a-spec", TINY_C51)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        C51MarketMaker(TINY_SPEC, "not-a-config")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        C51MarketMaker(TINY_SPEC, TINY_C51, action_grid=((0, 2),))


@requires_torch
def test_agent_act_and_q_values() -> None:
    agent = _tiny_agent(seed=1)
    rng = np.random.default_rng(0)
    s = _random_state(rng, agent.state_dim)
    q = agent.q_values(s)
    assert q.shape == (6,)
    assert np.all(np.isfinite(q))
    # Greedy act == argmax Q, repeatable, and does not advance the schedule.
    a1 = agent.act(s, greedy=True)
    a2 = agent.act(s, greedy=True)
    assert a1 == a2 == int(np.argmax(q))
    assert agent.n_decisions == 0
    # Non-greedy acts advance the epsilon schedule; after eps_decay_steps the
    # floor is reached.
    for _ in range(260):
        a = agent.act(s)
        assert 0 <= a < 6
    assert agent.n_decisions == 260
    assert agent.epsilon == pytest.approx(0.1)
    with pytest.raises(ValueError):
        agent.act(np.zeros(agent.state_dim - 1))
    with pytest.raises(ValueError):
        agent.act(np.full(agent.state_dim, np.nan))


@requires_torch
def test_agent_learn_finite_losses_and_projection_stability() -> None:
    agent = _tiny_agent(seed=2)
    rng = np.random.default_rng(3)
    dim = agent.state_dim
    for i in range(128):
        s = _random_state(rng, dim)
        agent.store_transition(
            s,
            int(rng.integers(6)),
            float(rng.normal(0.0, 0.05)),
            _random_state(rng, dim),
            0.99 ** int(rng.integers(1, 8)),
            i == 127,
        )
    losses = []
    for _ in range(30):
        loss = agent.learn()
        assert loss is not None  # buffer above batch size
        assert math.isfinite(loss)
        assert loss >= 0.0  # cross-entropy against a projected distribution
        losses.append(loss)
    assert agent.n_updates == 30
    s = _random_state(rng, dim)
    assert np.all(np.isfinite(agent.q_values(s)))


@requires_torch
def test_agent_learn_below_batch_returns_none() -> None:
    agent = _tiny_agent(seed=3)
    assert agent.learn() is None  # empty buffer
    rng = np.random.default_rng(4)
    s = _random_state(rng, agent.state_dim)
    for _ in range(5):
        agent.store_transition(s, 0, 0.01, s, 0.99, False)
    assert agent.learn() is None  # still below batch_size=32
    assert agent.n_updates == 0


@requires_torch
def test_agent_n_step_folding_and_flush() -> None:
    agent = _tiny_agent(seed=4)  # n_step = 3
    rng = np.random.default_rng(5)
    dim = agent.state_dim
    s = _random_state(rng, dim)
    agent.store_transition(s, 0, 0.1, s, 0.99, False)
    assert agent.total_stored == 0  # window not full
    agent.store_transition(s, 1, 0.1, s, 0.99, False)
    assert agent.total_stored == 0
    agent.store_transition(s, 2, 0.1, s, 0.99, False)
    assert agent.total_stored == 1  # first 3-step window folded
    agent.store_transition(s, 3, 0.1, s, 0.99, True)  # done flushes suffixes
    # pending held 2 items after the popleft; +done -> 3 suffix windows fold
    assert agent.total_stored == 4
    agent.begin_episode()
    agent.store_transition(s, 0, 0.1, s, 0.99, False)
    agent.begin_episode()  # drops the pending window across the episode bound
    agent.store_transition(s, 0, 0.1, s, 0.99, False)
    agent.store_transition(s, 0, 0.1, s, 0.99, False)
    assert agent.total_stored == 4  # no window completed after the flush


@requires_torch
def test_agent_store_transition_fail_closed() -> None:
    agent = _tiny_agent(seed=5)
    rng = np.random.default_rng(6)
    dim = agent.state_dim
    s = _random_state(rng, dim)
    with pytest.raises(ValueError):
        agent.store_transition(s, 6, 0.0, s, 0.99, False)  # action out of range
    with pytest.raises(ValueError):
        agent.store_transition(s, -1, 0.0, s, 0.99, False)
    with pytest.raises(ValueError):
        agent.store_transition(s, 0, float("nan"), s, 0.99, False)
    with pytest.raises(ValueError):
        agent.store_transition(s, 0, 0.0, s, 1.5, False)  # gamma_eff > 1
    with pytest.raises(ValueError):
        agent.store_transition(s, 0, 0.0, s, 0.0, False)  # gamma_eff <= 0
    with pytest.raises(ValueError):
        agent.store_transition(np.zeros(dim + 1), 0, 0.0, s, 0.99, False)


@requires_torch
def test_agent_training_determinism_and_seed_sensitivity() -> None:
    """Seeded torch + numpy: identical budgets give bit-identical training."""
    budgets = dict(
        horizon=250.0,
        n_episodes=1,
        decision_interval=1.0,
        sample_interval=25.0,
        reward_phi=1e-3,
    )
    a1 = _tiny_agent(seed=7)
    out1 = train_c51_market_maker(agent=a1, config=santa_fe_config(seed=0), seed_base=21, **budgets)
    a2 = _tiny_agent(seed=7)
    out2 = train_c51_market_maker(agent=a2, config=santa_fe_config(seed=0), seed_base=21, **budgets)
    a3 = _tiny_agent(seed=8)
    out3 = train_c51_market_maker(agent=a3, config=santa_fe_config(seed=0), seed_base=21, **budgets)
    assert out1["loss_curve"] == out2["loss_curve"]
    assert out1["episodes"] == out2["episodes"]
    assert out1["n_updates"] == out2["n_updates"]
    assert out1["loss_curve"] != out3["loss_curve"]
    assert out1["episodes"][0]["sim_internal_mtm_pnl_final"] == pytest.approx(
        out2["episodes"][0]["sim_internal_mtm_pnl_final"]
    )


@requires_torch
def test_agent_session_fail_closed_edges() -> None:
    agent = _tiny_agent(seed=6)  # spec cap = CAP = 12
    cfg = santa_fe_config(seed=1)
    with pytest.raises(ValueError):
        run_rl_mm_session(config=cfg, horizon=100.0, agent=agent)  # cap required
    with pytest.raises(ValueError):
        run_rl_mm_session(
            config=cfg, horizon=100.0, agent=agent, inventory_cap=CAP + 1
        )  # cap mismatch
    with pytest.raises(TypeError):
        run_rl_mm_session(
            config=cfg,
            horizon=100.0,
            agent="not-an-agent",
            inventory_cap=CAP,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        train_c51_market_maker(
            agent="not-an-agent",  # type: ignore[arg-type]
            config=cfg,
            horizon=100.0,
        )


# ---------------------------------------------------------------------------
# Tiny-budget end-to-end: train, then evaluate vs AS / GLFT
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> tuple[C51MarketMaker, dict]:
    """Train a tiny C51 agent on stationary flow (module-scoped, computed once)."""
    if not _HAS_TORCH:  # pragma: no cover - fixture only used by torch tests
        pytest.skip("torch not installed")
    agent = _tiny_agent(seed=0)
    out = train_c51_market_maker(
        agent=agent,
        config=santa_fe_config(seed=0),
        horizon=TRAIN_HORIZON,
        n_episodes=TRAIN_EPISODES,
        seed_base=11,
        decision_interval=1.0,
        sample_interval=25.0,
        reward_phi=1e-3,
    )
    return agent, out


@pytest.fixture(scope="module")
def eval_bundle(trained: tuple[C51MarketMaker, dict]) -> dict:
    """Paired-seed C51-vs-AS-vs-GLFT evaluation, stationary + regime flow."""
    agent, _ = trained
    return evaluate_rl_market_makers(
        agent=agent,
        config=santa_fe_config(seed=0),
        horizon=EVAL_HORIZON,
        n_seeds=1,
        seed_base=501,
        decision_interval=1.0,
        sample_interval=25.0,
        regime_tau_r=30.0,
        regime_omega=0.30,
    )


@requires_torch
def test_tiny_training_completes_with_bounded_inventory(
    trained: tuple[C51MarketMaker, dict],
) -> None:
    """Plumbing assertion: trained policy runs full sessions, inventory stays
    inside the hard cap. NOT a claim that the policy beats GLFT at this
    budget."""
    agent, out = trained
    assert out["label"] == "SYNTHETIC"
    assert out["kind"] == "c51_training_run"
    assert out["flow"] == "stationary"
    assert out["research_only"] is True
    assert out["live_pnl_claim"] is False
    assert out["n_episodes"] == TRAIN_EPISODES
    assert len(out["episodes"]) == TRAIN_EPISODES
    for ep in out["episodes"]:
        assert ep["session_completed"] is True
        assert ep["max_abs_inventory"] <= CAP  # hard bound never violated
        assert ep["n_decisions"] > 100
        assert math.isfinite(ep["sim_internal_mtm_pnl_final"])
    losses = np.asarray(out["loss_curve"], dtype=np.float64)
    assert losses.size > 100
    assert np.all(np.isfinite(losses))
    assert np.all(losses >= 0.0)
    assert out["n_updates"] > 0
    assert out["n_transitions_stored"] > 0
    assert out["epsilon_final"] == pytest.approx(TINY_C51.eps_end)
    assert agent.n_decisions > TRAIN_EPISODES * 100


@requires_torch
def test_training_session_identity_accounting() -> None:
    """Per-session bookkeeping identities for the RL runner."""
    agent = _tiny_agent(seed=9)
    out = run_rl_mm_session(
        config=santa_fe_config(seed=5),
        horizon=250.0,
        agent=agent,
        training=True,
        decision_interval=1.0,
        sample_interval=25.0,
        inventory_cap=CAP,
    )
    assert out["policy_kind"] == "c51_rl"
    assert out["training"] is True
    assert out["session_completed"] is True
    hist = out["action_histogram"]
    assert hist is not None and len(hist) == 6
    acted = sum(hist)
    # Every non-skipped decision acts exactly once; every acted decision is
    # eventually settled as one reward observation (terminal flush included).
    assert acted == out["n_decisions"] - out["n_skipped_decisions"]
    assert len(out["sim_internal_reward_path"]) == acted
    assert out["n_learn_steps"] >= 1
    assert all(math.isfinite(v) for v in out["loss_curve"])
    assert out["max_abs_inventory"] <= CAP
    # Auxiliary filter consumed exactly the executed MO stream (= sign stream).
    assert out["filter_n_mo"] == out["n_signs"]
    belief = out["filter_belief_final"]
    assert belief is not None and len(belief) == 2
    assert -1.0 <= belief[0] <= 1.0
    assert belief[1] >= 0.0
    # Safety layer + gating counters are present and non-negative.
    assert out["n_quote_clips"] >= 0
    assert out["n_inventory_gated"] >= 0
    assert out["epsilon_initial"] >= out["epsilon_final"]


@requires_torch
def test_greedy_eval_does_not_mutate_agent(trained: tuple[C51MarketMaker, dict]) -> None:
    agent, _ = trained
    n_dec, n_upd, stored = agent.n_decisions, agent.n_updates, agent.total_stored
    out = run_rl_mm_session(
        config=santa_fe_config(seed=77),
        horizon=200.0,
        agent=agent,
        training=False,
        decision_interval=1.0,
        inventory_cap=CAP,
    )
    assert out["session_completed"] is True
    assert out["loss_curve"] == []
    assert agent.n_decisions == n_dec  # greedy acts are not counted decisions
    assert agent.n_updates == n_upd
    assert agent.total_stored == stored


@requires_torch
def test_eval_harness_structure_and_metrics(eval_bundle: dict) -> None:
    b = eval_bundle
    assert b["label"] == "SYNTHETIC"
    assert b["kind"] == "c51_vs_classic_evaluation"
    assert b["research_only"] is True
    assert b["live_pnl_claim"] is False
    assert b["claim"] == "simulator_internal_diagnostic_only"
    assert b["policies"] == ("c51", "as", "glft")
    assert b["flows"] == ("stationary", "regime")
    combos = [f"{p}_{f}" for p in b["policies"] for f in b["flows"]]
    assert sorted(b["sessions"]) == sorted(combos)
    metric_suffixes = (
        "session_completion_rate",
        "inventory_saturation_rate",
        "max_abs_inventory_mean",
        "mean_abs_inventory_mean",
        "inventory_final_abs_mean",
        "n_fills_mean",
        "n_decisions_mean",
        "n_inventory_gated_mean",
        "mean_spread_ticks_mean",
        "regime_detected_rate",
    )
    pnl_suffixes = (
        "final_mean",
        "final_std",
        "path_mean",
        "path_std",
        "path_min",
    )
    for combo in combos:
        for suffix in metric_suffixes:
            key = f"{combo}_{suffix}"
            assert key in b["metrics"], f"missing metric {key}"
            assert math.isfinite(b["metrics"][key]), f"non-finite metric {key}"
        for suffix in pnl_suffixes:
            key = f"sim_internal_mtm_pnl_{suffix}_{combo}"
            assert key in b["metrics"], f"missing metric {key}"
            assert math.isfinite(b["metrics"][key]), f"non-finite metric {key}"
        rows = b["sessions"][combo]
        assert len(rows) == b["n_seeds"]
        for row in rows:
            assert row["session_completed"] is True
            assert row["max_abs_inventory"] <= CAP
            assert row["n_decisions"] > 50
            assert math.isfinite(row["sim_internal_mtm_pnl_final"])
            assert row["mean_spread_ticks"] >= 1.0
            assert row["phase"] in ("orderly_tight", "intermediate", "disordered_wide")
    # Gap identities: contrast keys equal the difference of the per-policy means.
    for flow in ("stationary", "regime"):
        for other in ("glft", "as"):
            gap_key = f"sim_internal_mtm_pnl_gap_c51_minus_{other}_{flow}_mean"
            assert gap_key in b["metrics"]
            expected = (
                b["metrics"][f"sim_internal_mtm_pnl_final_mean_c51_{flow}"]
                - b["metrics"][f"sim_internal_mtm_pnl_final_mean_{other}_{flow}"]
            )
            assert b["metrics"][gap_key] == pytest.approx(expected)
    # All policies completed every session (saturation may still differ).
    for combo in combos:
        assert b["metrics"][f"{combo}_session_completion_rate"] == pytest.approx(1.0)


@requires_torch
def test_eval_policies_are_distinct_actors(eval_bundle: dict) -> None:
    """At the pinned tiny seeds the three policies do not trace identical
    sessions (they differ in fills, inventory, or internal MTM)."""
    b = eval_bundle
    for flow in ("stationary", "regime"):
        rows = {p: b["sessions"][f"{p}_{flow}"][0] for p in ("c51", "as", "glft")}
        signatures = {
            (r["n_fills"], r["max_abs_inventory"], round(r["sim_internal_mtm_pnl_final"], 9))
            for r in rows.values()
        }
        assert len(signatures) >= 2, f"policies indistinguishable under {flow} flow"


@requires_torch
def test_honesty_no_forbidden_headline_keys(
    trained: tuple[C51MarketMaker, dict], eval_bundle: dict
) -> None:
    _, train_out = trained
    bundles = [
        train_out,
        eval_bundle,
        run_rl_mm_session(
            config=santa_fe_config(seed=9),
            horizon=200.0,
            policy=_as_pol(),
            inventory_cap=CAP,
        ),
    ]
    for bundle in bundles:
        keys = _all_keys(bundle)
        low = [k.lower() for k in keys]
        for tok in FORBIDDEN_HEADLINE_TOKENS:
            assert not any(tok in k for k in low), f"forbidden token {tok!r} in keys"
        for k in low:
            if "pnl" in k:
                assert k == "live_pnl_claim" or k.startswith("sim_internal_"), (
                    f"pnl metric key not simulator-internal: {k}"
                )
        assert bundle["research_only"] is True
        assert bundle["live_pnl_claim"] is False
        assert bundle["claim"] == "simulator_internal_diagnostic_only"
        assert bundle["label"] == "SYNTHETIC"
        assert bundle["data_source"] == "SYNTHETIC_ZI_LOB_v1"


# ---------------------------------------------------------------------------
# Documented-optional full comparison (bench battery; excluded from PR gate)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_torch
def test_rl_mm_benchmark_scaled_plumbing() -> None:
    """Scaled-down ``rl_mm_benchmark`` run: the full C51-vs-AS-vs-GLFT
    comparison pipeline under regime-switching training flow. Structural
    assertions only — the paper-scale frontier comparison is the documented
    bench-battery use of ``rl_mm_benchmark``, not a PR-gate claim."""
    out = rl_mm_benchmark(
        config=santa_fe_config(seed=0),
        horizon=1000.0,
        train_episodes=3,
        eval_seeds=1,
        decision_interval=1.0,
        sample_interval=25.0,
        reward_phi=1e-3,
        regime_tau_r=30.0,
        regime_omega=0.30,
        state_spec=RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=CAP),
        agent_config=C51Config(
            n_atoms=31,
            hidden=(64,),
            gamma_event=0.999,
            n_step=3,
            lr=1e-3,
            batch_size=32,
            buffer_capacity=8192,
            target_update_every=50,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=1200,
            seed=3,
        ),
        train_under_regime=True,
        seed=3,
    )
    assert out["label"] == "SYNTHETIC"
    assert out["kind"] == "c51_benchmark"
    assert out["train_under_regime"] is True
    assert out["bench_keys"] == sorted(out["bench_keys"])
    assert len(out["bench_keys"]) >= 90  # 6 combos x 15 metrics + 4 gap keys
    train = out["training"]
    assert train["flow"] == "regime_factory"
    assert len(train["episodes"]) == 3
    for ep in train["episodes"]:
        assert ep["session_completed"] is True
        assert ep["max_abs_inventory"] <= CAP
        assert math.isfinite(ep["sim_internal_mtm_pnl_final"])
    assert all(math.isfinite(v) for v in train["loss_curve"])
    ev = out["evaluation"]
    for combo, rows in ev["sessions"].items():
        assert rows and combo.split("_", 1)[0] in ("c51", "as", "glft")
        for row in rows:
            assert row["session_completed"] is True
            assert row["max_abs_inventory"] <= CAP
    keys = _all_keys(out)
    low = [k.lower() for k in keys]
    for tok in FORBIDDEN_HEADLINE_TOKENS:
        assert not any(tok in k for k in low)
    for k in low:
        if "pnl" in k:
            assert k == "live_pnl_claim" or k.startswith("sim_internal_")
