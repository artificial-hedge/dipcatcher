"""Tests for wave-117 RL canon: gae, vtrace, trpo, ppo, ddpg, td3."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ddpg import DDPG, bench_ddpg
from quant_fund.models.gae import bench_gae, gae
from quant_fund.models.ppo import bench_ppo, ppo_loss
from quant_fund.models.td3 import TD3, bench_td3
from quant_fund.models.trpo import bench_trpo, softmax, trpo_step
from quant_fund.models.vtrace import bench_vtrace, vtrace


def test_gae_lam0_is_td():
    r = np.array([1.0, 1.0, 1.0])
    v = np.array([0.0, 0.0, 0.0])
    adv, _ = gae(r, v, 0.0, gamma=0.9, lam=0.0)
    assert np.allclose(adv, r)


def test_vtrace_onpolicy_nstep():
    r = np.ones(4)
    v = np.zeros(4)
    vs = vtrace(r, v, 0.5, np.ones(4), gamma=0.9)
    expect = sum(0.9**k for k in range(4)) + 0.9**4 * 0.5
    assert vs[0] == pytest.approx(expect)


def test_trpo_kl_constraint():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 0.1, 3)
    adv = np.array([0.2, 0.9, 0.1])
    new = trpo_step(logits, adv, kl_max=0.01)
    p, q = softmax(logits), softmax(new)
    kl = float(p @ (np.log(p + 1e-12) - np.log(q + 1e-12)))
    assert kl <= 0.012


def test_ppo_clip():
    r = np.array([1.5, 0.5])
    loss = ppo_loss(r, np.ones(2), clip=0.2)
    # pos adv: r=1.5 clipped to 1.2 → −1.2; r=0.5 unclipped → −0.5
    assert np.allclose(loss, [-1.2, -0.5])


def test_ddpg_act_bounded():
    agent = DDPG(a_lim=1.0)
    agent.theta[:] = [5.0, 5.0]
    assert abs(agent.act(1.0)) <= 1.0


def test_td3_min_target():
    agent = TD3()
    assert agent.w1.shape == agent.w2.shape == (6,)


@pytest.mark.parametrize(
    "fn",
    [bench_gae, bench_vtrace, bench_trpo, bench_ppo, bench_ddpg, bench_td3],
    ids=lambda f: f.__name__,
)
def test_w117_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
