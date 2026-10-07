"""Quantile-regression DQN (Dabney et al. 2018) (SYNTHETIC).

32 quantiles trained with the Huber quantile loss; CVaR action selection on
the learned quantile grid. Same risk-sensitive niche as C51 with a
nonparametric quantile representation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import cvar, mc_return_dist, sample_return, synth_reward_env

FloatArray = NDArray[np.float64]

_NQ = 32


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("qr_dqn needs the torch `nn` extra") from exc


def _huber(u, kappa: float = 1.0):
    import torch

    au = u.abs()
    return torch.where(au <= kappa, 0.5 * u * u, kappa * (au - 0.5 * kappa))


def bench_qr_dqn(
    seed: int = 13,
    n_train: int = 600,
    iters: int = 400,
    batch: int = 64,
    mc_eval: int = 2000,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    taus = (torch.arange(_NQ) + 0.5) / _NQ
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 48),
        torch.nn.ReLU(),
        torch.nn.Linear(48, 2 * _NQ),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _it in range(iters):
        idx = rng.integers(0, n_train, batch)
        xb = xt[idx]
        qs = net(xb).view(-1, 2, _NQ)
        a_take = rng.integers(0, 2, batch)
        rew = sample_return(s[idx], a_take.astype(float), rng)
        with torch.no_grad():
            tgt = torch.tensor(rew).float()[:, None]
        q_sel = qs[np.arange(batch), a_take]
        u = tgt - q_sel
        loss = (_huber(u) * (taus - (u < 0).float()).abs()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q_all = net(xt).view(-1, 2, _NQ).numpy()
    cvar_a = q_all[:, :, : _NQ // 10 + 1].mean(-1)
    per_state = np.zeros((4, 2))
    mean_q = np.zeros((4, 2))
    for st in range(4):
        mask = s == st
        per_state[st] = cvar_a[mask].mean(0)
        mean_q[st] = q_all[mask].mean(-1).mean(0)
    a_risk = per_state.argmax(1)
    a_mean = mean_q.argmax(1)
    eval_rng = np.random.default_rng(seed + 1)
    cvar_risk = np.mean(
        [cvar(mc_return_dist(st, int(a_risk[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    cvar_mean = np.mean(
        [cvar(mc_return_dist(st, int(a_mean[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    pb = 0.0
    for st in range(4):
        mask = s == st
        for a in range(2):
            truth = np.quantile(mc_return_dist(st, a, eval_rng, 800), taus.numpy())
            pred = q_all[mask, a].mean(0)
            pb += np.abs(truth - pred).mean()
    return {
        "synthetic_qr_cvar": float(cvar_risk),
        "synthetic_qr_mean_cvar": float(cvar_mean),
        "synthetic_qr_cvar_gain": float(cvar_risk - cvar_mean),
        "synthetic_qr_quantile_mae": float(pb / 8),
        "synthetic_torch_available": 1.0,
    }
