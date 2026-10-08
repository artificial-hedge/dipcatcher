"""Exec-RL agent honesty tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.exec_rl import (
    ExecState,
    MarketSim,
    QExecAgent,
    bench_exec_rl,
    run_episode,
    train_q_agent,
)


def _sim() -> MarketSim:
    return MarketSim(steps=8)


def test_eval_does_not_mutate_q():
    """Held-out episodes must not keep training the agent."""
    sim = _sim()
    agent = train_q_agent(sim, 30, np.random.default_rng(0))
    assert agent.training is False
    q_before = agent.q.copy()
    rng = np.random.default_rng(1)
    for _ in range(5):
        run_episode(agent, sim.episode(rng), sim)
    np.testing.assert_array_equal(agent.q, q_before)


def test_agents_have_independent_rng():
    """Two fresh agents must not share a class-level RNG stream."""
    a1 = QExecAgent(8)
    a2 = QExecAgent(8)
    # consume draws on a1; a2's stream must be untouched
    for _ in range(10):
        a1.rng.random()
    fresh = np.random.default_rng(0)
    assert a2.rng.random() == fresh.random()


def test_episode_resets_stale_momentum():
    """Momentum from a previous episode must not leak into step 0."""
    sim = _sim()
    agent = QExecAgent(sim.steps)
    agent.training = False
    agent._mom = 1.0  # stale: positive momentum left over
    captured: list[int] = []
    orig = agent._state

    def spy(s: ExecState):
        st = orig(s)
        captured.append(st[2])
        return st

    agent._state = spy  # type: ignore[method-assign]
    run_episode(agent, sim.episode(np.random.default_rng(3)), sim)
    assert captured and captured[0] == 0


def test_reset_clears_pending_transition():
    agent = QExecAgent(8)
    agent._s = (0, 0, 0)
    agent._mom = 0.5
    agent._reward = 1.5
    agent.reset()
    assert agent._s is None and agent._mom == 0.0 and agent._reward == 0.0


def test_bench_exec_rl_runs():
    out = bench_exec_rl(seed=7)
    for k in (
        "synthetic_exec_rl_agent_is_bps",
        "synthetic_exec_rl_twap_is_bps",
        "synthetic_exec_rl_margin_bps",
        "synthetic_exec_rl_agent_beats_twap_rate",
    ):
        assert np.isfinite(out[k])
