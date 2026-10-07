"""Implicit Quantile Network (Dabney et al. 2018) (SYNTHETIC).

Quantile conditioned on sampled τ via a cosine embedding; risk measures
(CVaR-α distortion) computed by reweighting sampled quantiles rather than
committing to a fixed grid — the IQN advantage for risk-sensitive control.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import cvar, mc_return_dist, sample_return, synth_reward_env

FloatArray = NDArray[np.float64]

_NT = 8
_NP = 32


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("iqn_dqn needs the torch `nn` extra") from exc


def _phi(torch, tau):
    i = torch.arange(64, dtype=tau.dtype)
    return torch.cos(torch.pi * i * tau)


def bench_iqn_dqn(
    seed: int = 17,
    n_train: int = 600,
    iters: int = 400,
    batch: int = 48,
    mc_eval: int = 2000,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    emb = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU())
    head = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=3e-3)
    for _it in range(iters):
        idx = rng.integers(0, n_train, batch)
        xb = xt[idx]
        z = emb(xb)
        taus = torch.rand(batch, _NT, 1)
        pe = _phi(torch, taus)
        zq = (z[:, None, :] * pe).view(batch * _NT, 64)
        qs = head(zq).view(batch, _NT, 2)
        a_take = rng.integers(0, 2, batch)
        rew = torch.tensor(sample_return(s[idx], a_take.astype(float), rng))
        q_sel = torch.gather(qs, 2, torch.tensor(a_take)[:, None, None].expand(-1, _NT, 1)).squeeze(
            -1
        )
        u = rew[:, None] - q_sel
        taus_b = taus[:, :, 0]
        hub = torch.where(u.abs() <= 1.0, 0.5 * u * u, u.abs() - 0.5)
        loss = (hub * (taus_b - (u < 0).float()).abs()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        taus_e = (torch.arange(_NP) + 0.5)[:, None] / _NP
        pe = _phi(torch, taus_e)
        q_all = np.zeros((n_train, _NP, 2))
        for i in range(n_train):
            z = emb(xt[i : i + 1])
            q_all[i] = head(z.repeat(_NP, 1) * pe).numpy()
    cvar_q = q_all[:, : _NP // 10 + 1, :].mean(1)
    per_state = np.zeros((4, 2))
    mean_q = np.zeros((4, 2))
    for st in range(4):
        mask = s == st
        per_state[st] = cvar_q[mask].mean(0)
        mean_q[st] = q_all[mask].mean(1).mean(0)
    a_risk = per_state.argmax(1)
    a_mean = mean_q.argmax(1)
    eval_rng = np.random.default_rng(seed + 1)
    cvar_risk = np.mean(
        [cvar(mc_return_dist(st, int(a_risk[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    cvar_mean = np.mean(
        [cvar(mc_return_dist(st, int(a_mean[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    return {
        "synthetic_iqn_cvar": float(cvar_risk),
        "synthetic_iqn_mean_cvar": float(cvar_mean),
        "synthetic_iqn_cvar_gain": float(cvar_risk - cvar_mean),
        "synthetic_iqn_risk_share": float(np.mean(a_risk == 0)),
        "synthetic_torch_available": 1.0,
    }
