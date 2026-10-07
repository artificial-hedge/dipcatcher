"""Prioritized experience replay (Schaul et al. 2016) (SYNTHETIC).

TD-error proportional priorities (α=0.6, IS weights β→1) over a small
replay buffer; on a rare-outcome env where the heavy-left-tail action's
downside is sparse, PER reaches the risk-aware policy in fewer gradient
steps than uniform replay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import mc_return_dist, sample_return, synth_reward_env

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("prioritized_replay needs the torch `nn` extra") from exc


def _train(
    torch,
    rng: np.random.Generator,
    n_train: int,
    iters: int,
    batch: int,
    prioritized: bool,
) -> FloatArray:
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    prio = np.ones(n_train)
    for _it in range(iters):
        if prioritized:
            p = prio**0.6
            p = p / p.sum()
            idx = rng.choice(n_train, batch, p=p)
            w = (n_train * p[idx]) ** (-0.4)
            w = w / w.max()
        else:
            idx = rng.integers(0, n_train, batch)
            w = np.ones(batch)
        a_take = rng.integers(0, 2, batch)
        rew = sample_return(s[idx], a_take.astype(float), rng)
        q = net(xt[idx])
        td = torch.tensor(rew).float() - q[np.arange(batch), a_take]
        loss = (torch.tensor(w).float() * td.pow(2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if prioritized:
            prio[idx] = np.abs(td.detach().numpy()) + 1e-3
    with torch.no_grad():
        return np.asarray(net(torch.eye(4)).numpy())


def bench_prioritized_replay(
    seed: int = 23,
    n_train: int = 300,
    iters: int = 200,
    batch: int = 32,
    mc_eval: int = 2000,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q_per = _train(torch, rng, n_train, iters, batch, True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q_uni = _train(torch, rng, n_train, iters, batch, False)
    eval_rng = np.random.default_rng(seed + 1)
    truth = np.zeros(4)
    for st in range(4):
        truth[st] = mc_return_dist(st, 1, eval_rng, 3000).mean()

    def _tail_mae(q: FloatArray) -> float:
        a_hat = q[:, 1]
        return float(np.abs(a_hat - truth).mean())

    mae_per = _tail_mae(q_per)
    mae_uni = _tail_mae(q_uni)
    return {
        "synthetic_per_tail_mae": mae_per,
        "synthetic_per_uniform_mae": mae_uni,
        "synthetic_per_gain": mae_uni - mae_per,
        "synthetic_torch_available": 1.0,
    }
