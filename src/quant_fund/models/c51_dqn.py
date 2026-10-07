"""C51 categorical distributional DQN (Bellemare et al. 2017) (SYNTHETIC).

Q-value distribution over a fixed 51-atom support; Bellman projection onto
the categorical atoms. On the risk-sensitive env the CVaR-optimal action is
chosen from the learned return distribution, beating the mean-only greedy
policy that is indifferent between actions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import mc_return_dist, sample_return, synth_reward_env

FloatArray = NDArray[np.float64]

_ATOMS = 51
_VMIN, _VMAX = -4.0, 2.0


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("c51_dqn needs the torch `nn` extra") from exc


def _project_delta(torch, rewards: FloatArray, support: FloatArray) -> object:
    """Project scalar reward deltas onto the categorical atoms."""
    dz = support[1] - support[0]
    b = (np.clip(rewards, _VMIN, _VMAX) - _VMIN) / dz
    lo = np.floor(b).astype(int)
    hi = np.minimum(lo + 1, _ATOMS - 1)
    m = torch.zeros(rewards.size, _ATOMS)
    for i in range(rewards.size):
        m[i, lo[i]] += hi[i] - b[i]
        m[i, hi[i]] += b[i] - lo[i]
    return m


def bench_c51_dqn(
    seed: int = 11,
    n_train: int = 600,
    iters: int = 800,
    batch: int = 64,
    mc_eval: int = 2000,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z_atoms = np.linspace(_VMIN, _VMAX, _ATOMS)
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 48),
        torch.nn.ReLU(),
        torch.nn.Linear(48, 2 * _ATOMS),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _it in range(iters):
        idx = rng.integers(0, n_train, batch)
        xb = xt[idx]
        logits = net(xb).view(-1, 2, _ATOMS)
        a_take = rng.integers(0, 2, batch)
        rew = sample_return(s[idx], a_take.astype(float), rng)
        tgt = _project_delta(torch, rew, z_atoms)
        loss = -(tgt * torch.log_softmax(logits[np.arange(batch), a_take], -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(net(xt).view(-1, 2, _ATOMS), -1).numpy()
    zt = z_atoms
    w1_c51 = 0.0
    w1_gauss = 0.0
    tail_pred = np.zeros(2)
    tail_gauss = np.zeros(2)
    risk_pick = np.zeros(4)
    eval_rng = np.random.default_rng(seed + 1)
    for a in range(2):
        truth = mc_return_dist(0, a, eval_rng, 4000)
        pred_p = probs[:, a, :].mean(0)
        pred_cdf = np.cumsum(pred_p)
        truth_cdf = np.searchsorted(np.sort(truth), zt) / truth.size
        w1_c51 += float(np.trapezoid(np.abs(pred_cdf - truth_cdf), zt))
        mu, sd = float(truth.mean()), float(truth.std() + 1e-9)
        from math import erf

        gauss_cdf = np.array([0.5 * (1 + erf((z - mu) / (sd * 2**0.5))) for z in zt])
        w1_gauss += float(np.trapezoid(np.abs(gauss_cdf - truth_cdf), zt))
        tail_pred[a] = pred_p[zt < -1.0].sum()
        tail_gauss[a] = float(1 - gauss_cdf[np.searchsorted(zt, -1.0)])
    for st in range(4):
        mask = s == st
        tail_state = (
            probs[mask].mean(0)[1][zt < -1.0].sum() - probs[mask].mean(0)[0][zt < -1.0].sum()
        )
        risk_pick[st] = float(tail_state > 0)
    return {
        "synthetic_c51_w1": float(w1_c51 / 2),
        "synthetic_c51_gauss_w1": float(w1_gauss / 2),
        "synthetic_c51_w1_gain": float((w1_gauss - w1_c51) / 2),
        "synthetic_c51_risk_share": float(risk_pick.mean()),
        "synthetic_c51_tail_pred": float(tail_pred[1]),
        "synthetic_c51_tail_gauss": float(tail_gauss[1]),
        "synthetic_torch_available": 1.0,
    }
