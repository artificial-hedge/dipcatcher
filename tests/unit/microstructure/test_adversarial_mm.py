"""Tests for adversarial_mm — the learned-adversary semi-MDP lane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.adversarial_mm import (
    ADVERSARIAL_MM_REVISION,
    P_BUY_GRID,
    RandomAdversary,
    adversarial_episode,
    constant_flow_sweep,
    make_pareto_duration,
    make_uniform_duration,
    run_adversarial_eval,
)
from quant_fund.microstructure.zi_lob_simulator import (
    AdversarialFlow,
    glft_policy,
    santa_fe_config,
)

TINY_CONFIG = santa_fe_config(seed=7)
POL_KW = {"gamma": 1.0, "kappa": 1000.0, "sigma": 0.02, "a_fill": 1.0, "tick": 0.01}


def _const_picker(p: float):
    def _pick(inv: float, prev: float) -> float:
        del inv
        assert 0.0 < prev < 1.0
        return p

    return _pick


# ---------------------------------------------------------------------------
# AdversarialFlow contract
# ---------------------------------------------------------------------------


def test_flow_fail_closed_construction() -> None:
    with pytest.raises(TypeError):
        AdversarialFlow(picker="x", duration_sampler=make_uniform_duration(5), seed=0)
    with pytest.raises(TypeError):
        AdversarialFlow(picker=_const_picker(0.5), duration_sampler=42, seed=0)
    with pytest.raises(ValueError):
        AdversarialFlow(
            picker=_const_picker(0.5),
            duration_sampler=make_uniform_duration(5),
            seed=0,
            first_p_buy=1.5,
        )
    with pytest.raises(ValueError):
        AdversarialFlow(
            picker=_const_picker(0.5),
            duration_sampler=make_uniform_duration(5),
            seed=0,
            max_legs=0,
        )
    with pytest.raises(ValueError):
        AdversarialFlow(
            picker=_const_picker(0.5),
            duration_sampler=lambda rng: 0,
            seed=0,
        )


def test_flow_extension_and_boundary_log() -> None:
    seen: list[tuple[float, float]] = []

    def picker(inv: float, prev: float) -> float:
        seen.append((inv, prev))
        return 0.7

    flow = AdversarialFlow(
        picker=picker, duration_sampler=make_uniform_duration(3), seed=0, first_p_buy=0.4
    )
    flow.note_inventory(5.0)
    assert flow.current().p_buy == 0.4
    for _ in range(3):
        flow.advance()
    assert flow.n_mo == 3
    # Boundary fires on the next current() — picker sees the noted inventory.
    assert flow.current().p_buy == 0.7
    assert seen == [(5.0, 0.4)]
    assert flow.boundary_log == [(3, 5.0, 0.4, 0.7)]
    for _ in range(3):
        flow.advance()
    st = flow.current()
    assert st.p_buy == 0.7 and flow.n_legs == 3
    assert flow.state_mo_counts == [3, 3, 0]


def test_flow_expected_p_buy_and_fail_closed() -> None:
    flow = AdversarialFlow(
        picker=_const_picker(0.6),
        duration_sampler=make_uniform_duration(4),
        seed=0,
        first_p_buy=0.2,
    )
    with pytest.raises(ValueError):
        flow.expected_p_buy()
    with pytest.raises(ValueError):
        flow.note_inventory(float("nan"))
    for _ in range(5):
        # the engine calls current() on every step before advance()
        flow.current()
        flow.advance()
    assert flow.expected_p_buy() == pytest.approx((4 * 0.2 + 1 * 0.6) / 5)


def test_flow_picker_fail_closed() -> None:
    flow = AdversarialFlow(
        picker=lambda inv, prev: 1.2,
        duration_sampler=make_uniform_duration(2),
        seed=0,
    )
    for _ in range(2):
        flow.advance()
    with pytest.raises(RuntimeError):
        flow.current()


def test_flow_max_legs_fail_closed() -> None:
    flow = AdversarialFlow(
        picker=_const_picker(0.6),
        duration_sampler=make_uniform_duration(1),
        seed=0,
        max_legs=2,
    )
    with pytest.raises(RuntimeError):
        for _ in range(3):
            flow.advance()
            flow.current()


def test_flow_determinism() -> None:
    def build() -> AdversarialFlow:
        picks = iter([0.7, 0.3, 0.7])
        return AdversarialFlow(
            picker=lambda i, p: next(picks),
            duration_sampler=make_pareto_duration(alpha=1.5, l_min=2, l_cap=10),
            seed=42,
        )

    f1, f2 = build(), build()
    for _ in range(12):
        f1.advance()
        f2.advance()
        f1.current()
        f2.current()
    assert [n for _, n in f1._legs] == [n for _, n in f2._legs]


def test_duration_samplers_fail_closed() -> None:
    with pytest.raises(ValueError):
        make_pareto_duration(alpha=0.0)
    with pytest.raises(ValueError):
        make_pareto_duration(l_min=0)
    with pytest.raises(ValueError):
        make_pareto_duration(l_min=5, l_cap=3)
    with pytest.raises(ValueError):
        make_uniform_duration(0)
    rng = np.random.default_rng(0)
    d = make_pareto_duration(alpha=1.5, l_min=3, l_cap=10)
    assert all(3 <= d(rng) <= 10 for _ in range(200))


# ---------------------------------------------------------------------------
# RandomAdversary (torch-free)
# ---------------------------------------------------------------------------


def test_random_adversary() -> None:
    adv = RandomAdversary(seed=0)
    idx = [adv.select(0.0, 0.5) for _ in range(50)]
    assert all(0 <= i < len(P_BUY_GRID) for i in idx)
    assert len(set(idx)) > 3
    with pytest.raises(ValueError):
        adv.select(0.0, 1.5)


def test_adversarial_episode_classic_defender() -> None:
    pol = glft_policy(**POL_KW)
    bundle, flow = adversarial_episode(
        config=TINY_CONFIG,
        horizon=300.0,
        picker=RandomAdversary(seed=1).picker(),
        duration_sampler=make_uniform_duration(4),
        seed=0,
        defender_policy=pol,
        inventory_cap=10,
    )
    assert bundle["label"] == "SYNTHETIC"
    assert bundle["n_events"] > 0
    assert flow.n_legs >= 2
    assert math.isfinite(bundle["sim_internal_mtm_pnl_final"])


def test_episode_determinism() -> None:
    pol = glft_policy(**POL_KW)

    def run() -> float:
        bundle, _ = adversarial_episode(
            config=TINY_CONFIG,
            horizon=200.0,
            picker=RandomAdversary(seed=3).picker(),
            duration_sampler=make_uniform_duration(4),
            seed=11,
            defender_policy=pol,
            inventory_cap=10,
        )
        return float(bundle["sim_internal_mtm_pnl_final"])

    assert run() == run()


def test_episode_fail_closed() -> None:
    with pytest.raises(ValueError):
        adversarial_episode(
            config=TINY_CONFIG,
            horizon=10.0,
            picker=RandomAdversary(seed=0).picker(),
            duration_sampler=make_uniform_duration(5),
            seed=0,
        )
    with pytest.raises(TypeError):
        adversarial_episode(
            config=TINY_CONFIG,
            horizon=10.0,
            picker=RandomAdversary(seed=0).picker(),
            duration_sampler=make_uniform_duration(5),
            seed=0,
            defender_policy="x",
        )


def test_constant_flow_sweep() -> None:
    pol = glft_policy(**POL_KW)
    out = constant_flow_sweep(
        config=TINY_CONFIG,
        horizon=150.0,
        defender_policy=pol,
        inventory_cap=10,
        p_buy_grid=(0.2, 0.5, 0.8),
        seed=0,
    )
    assert out["kind"] == "constant_flow_sweep"
    assert out["label"] == "SYNTHETIC"
    assert len(out["rows"]) == 3
    assert all(math.isfinite(r["sim_internal_terminal_score"]) for r in out["rows"])
    assert out["worst_p_buy"] in (0.2, 0.5, 0.8)
    # replay determinism
    out2 = constant_flow_sweep(
        config=TINY_CONFIG,
        horizon=150.0,
        defender_policy=pol,
        inventory_cap=10,
        p_buy_grid=(0.2, 0.5, 0.8),
        seed=0,
    )
    assert out["worst_score"] == out2["worst_score"]


def test_sweep_fail_closed() -> None:
    with pytest.raises(ValueError):
        constant_flow_sweep(
            config=TINY_CONFIG,
            horizon=10.0,
            defender_policy=glft_policy(**POL_KW),
            p_buy_grid=(),
        )
    with pytest.raises(ValueError):
        constant_flow_sweep(
            config=TINY_CONFIG,
            horizon=10.0,
            defender_policy=glft_policy(**POL_KW),
            p_buy_grid=(1.5,),
        )


# ---------------------------------------------------------------------------
# Torch-gated AdversaryAgent + full eval loop
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="needs the nn extra")


def test_adversary_agent_basics() -> None:
    from quant_fund.microstructure.adversarial_mm import AdversaryAgent

    adv = AdversaryAgent(seed=0, batch_size=4)
    idx = adv.select(0.0, 0.5)
    assert 0 <= idx < 13
    obs = adv.obs(12.0, 0.4)
    assert obs == (1.5, 0.4)
    for _ in range(8):
        adv.store_transition((0.0, 0.5), 0, 1.0, (0.1, 0.6), False)
    assert adv.learn() is not None
    eps0 = adv.epsilon
    adv.end_episode()
    assert adv.epsilon < eps0
    with pytest.raises(ValueError):
        adv.store_transition((0.0, 0.5), 13, 0.0, (0.0, 0.5), False)
    with pytest.raises(ValueError):
        adv.store_transition((0.0, 0.5), 0, float("nan"), (0.0, 0.5), False)


def test_adversary_agent_determinism() -> None:
    from quant_fund.microstructure.adversarial_mm import AdversaryAgent

    def build() -> AdversaryAgent:
        adv = AdversaryAgent(seed=5, eps_start=1.0)
        for i in range(6):
            adv.store_transition((0.1 * i, 0.5), i % 13, float(i), (0.1 * (i + 1), 0.5), False)
        return adv

    a1, a2 = build(), build()
    a1.learn()
    a2.learn()
    assert a1.losses == a2.losses


def test_run_adversarial_eval_policy() -> None:
    from quant_fund.microstructure.adversarial_mm import AdversaryAgent

    pol = glft_policy(**POL_KW)
    adv = AdversaryAgent(seed=0, batch_size=4)
    out = run_adversarial_eval(
        config=TINY_CONFIG,
        horizon=150.0,
        n_episodes=3,
        adversary=adv,
        defender_policy=pol,
        inventory_cap=10,
        duration_sampler=make_uniform_duration(4),
        seed=0,
    )
    assert out["kind"] == "adversarial_mm_eval"
    assert out["label"] == "SYNTHETIC"
    assert out["defender_kind"] == "classic_policy"
    assert len(out["episodes"]) == 3
    assert all(math.isfinite(e["sim_internal_terminal_score"]) for e in out["episodes"])
    assert out["adversary_n_updates"] > 0
    assert out["adversary_revision"] == ADVERSARIAL_MM_REVISION


def test_run_adversarial_eval_fail_closed() -> None:
    from quant_fund.microstructure.adversarial_mm import AdversaryAgent

    pol = glft_policy(**POL_KW)
    with pytest.raises(TypeError):
        run_adversarial_eval(
            config=TINY_CONFIG,
            horizon=10.0,
            n_episodes=1,
            adversary=RandomAdversary(seed=0),
            defender_policy=pol,
        )
    with pytest.raises(ValueError):
        run_adversarial_eval(
            config=TINY_CONFIG,
            horizon=10.0,
            n_episodes=0,
            adversary=AdversaryAgent(seed=0),
            defender_policy=pol,
        )


def test_adversary_reward_mass_conservation() -> None:
    """Sparse policy-defender mode: adversary return = -(pnl - penalty)."""
    from quant_fund.microstructure.adversarial_mm import AdversaryAgent

    pol = glft_policy(**POL_KW)
    adv = AdversaryAgent(seed=0, batch_size=4, eps_start=0.0, eps_end=0.0)
    out = run_adversarial_eval(
        config=TINY_CONFIG,
        horizon=120.0,
        n_episodes=2,
        adversary=adv,
        defender_policy=pol,
        inventory_cap=10,
        duration_sampler=make_uniform_duration(4),
        seed=1,
        train_adversary=False,
    )
    for e in out["episodes"]:
        assert e["adversary_return"] == pytest.approx(
            -(e["sim_internal_mtm_pnl_final"] - e["sim_internal_penalty_sum"])
        )


def test_run_adversarial_eval_c51_dense_rewards() -> None:
    """C51 defender: dense per-leg reward attribution via reward_mo_index."""
    from dataclasses import replace

    from quant_fund.microstructure.adversarial_mm import AdversaryAgent
    from quant_fund.microstructure.rl_market_maker import (
        C51Config,
        C51MarketMaker,
        RLStateSpec,
    )

    spec = RLStateSpec(aux_enabled=True, filter_tau_r=30.0, inventory_cap=12)
    c51 = C51Config(
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
        eps_start=0.0,
        eps_end=0.0,
        eps_decay_steps=250,
        seed=0,
    )
    agent = C51MarketMaker(spec, replace(c51, seed=0))
    adv = AdversaryAgent(seed=0, batch_size=4, inv_limit=12.0)
    out = run_adversarial_eval(
        config=TINY_CONFIG,
        horizon=120.0,
        n_episodes=2,
        adversary=adv,
        defender_agent=agent,
        inventory_cap=12,
        duration_sampler=make_uniform_duration(4),
        seed=2,
        train_adversary=False,
    )
    assert out["defender_kind"] == "c51_rl"
    # Exact accounting: adversary earns -defender-reward on legs it chose —
    # i.e. over MOs after the first boundary (the seed leg is unattributed).
    adv2 = AdversaryAgent(seed=0, batch_size=4, inv_limit=12.0)
    bundle, flow = adversarial_episode(
        config=replace(TINY_CONFIG, seed=9),
        horizon=120.0,
        picker=adv2.picker(greedy=True),
        duration_sampler=make_uniform_duration(4),
        seed=11,
        defender_agent=agent,
        inventory_cap=12,
    )
    rewards = bundle["sim_internal_reward_path"]
    mo_idx = bundle["sim_internal_reward_mo_index"]
    assert len(rewards) == len(mo_idx)
    first_bound = flow.boundary_log[0][0]
    attributed = -sum(r for r, m in zip(rewards, mo_idx, strict=True) if m > first_bound)
    assert all(b[0] > 0 for b in flow.boundary_log)
    assert isinstance(attributed, float)
    for e in out["episodes"]:
        assert math.isfinite(e["adversary_return"])
        assert e["n_boundaries"] >= 1
        assert len(e["boundary_mo"]) == e["n_boundaries"]
