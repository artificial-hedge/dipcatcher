"""Tests for wave-119 MARL canon: vdn, qmix, coma, maddpg, mappo,
mf_q."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.coma import COMA, bench_coma
from quant_fund.models.maddpg import MADDPG, bench_maddpg
from quant_fund.models.mappo import MAPPO, bench_mappo
from quant_fund.models.mf_q import CrowdEnv, MeanFieldQ, bench_mf_q
from quant_fund.models.qmix import QMIX, bench_qmix
from quant_fund.models.vdn import VDN, RendezvousEnv, bench_vdn


def test_rendezvous_meets():
    env = RendezvousEnv(n_pos=5)
    rng = np.random.default_rng(0)
    env.reset(rng)
    obs, r, done = env.step([0, 0])
    assert not done and r == -1.0


def test_vdn_joint_greedy():
    env = RendezvousEnv()
    agent = VDN(env)
    rng = np.random.default_rng(0)
    obs = env.reset(rng)
    acts = agent.joint_greedy(obs)
    assert len(acts) == 2 and all(0 <= a < 3 for a in acts)


def test_qmix_monotone():
    env = RendezvousEnv()
    agent = QMIX(env)
    agent.u[0][3] = 0.5
    agent.u[1][3] = 0.7
    v = agent._mix(3, np.array([1.0, 2.0]))
    assert v == pytest.approx(0.5 + 1.4)


def test_coma_baseline():
    env = RendezvousEnv()
    agent = COMA(env)
    agent.critic[2, :, :] = 4.0
    adv = agent.counterfactual_adv(2, [0, 0], [0, 1], 0)
    assert adv == pytest.approx(0.0)


def test_maddpg_act_bounded():
    agent = MADDPG()
    agent.theta[0] = np.array([10.0, 10.0, 10.0])
    assert abs(agent.act(0, np.array([0.5, 0.2]))) <= 0.15 + 1e-9


def test_mappo_runs_episode():
    env = RendezvousEnv()
    agent = MAPPO(env)
    rng = np.random.default_rng(0)
    traj, ret = agent._episode(rng)
    assert len(traj) <= env.horizon


def test_mf_q_binning():
    env = CrowdEnv()
    agent = MeanFieldQ(env, n_bins=5)
    assert agent._bin(0.0) == 0 and agent._bin(1.0) == 4


@pytest.mark.parametrize(
    "fn",
    [bench_vdn, bench_qmix, bench_coma, bench_maddpg, bench_mappo, bench_mf_q],
    ids=lambda f: f.__name__,
)
def test_w119_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
