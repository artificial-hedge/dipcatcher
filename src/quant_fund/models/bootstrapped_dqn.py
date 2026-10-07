"""Bootstrapped DQN posterior (Osband et al. 2016).

K=8 Q-heads trained on resampled data give an epistemic posterior over
Q; disagreement is exploitable for pessimistic risk-aware action choice.
Bench: posterior spread correlates with true prediction error, and the
lower-confidence-bound policy beats greedy on the risky env.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._drl_synth import cvar, mc_return_dist, sample_return, synth_reward_env

FloatArray = NDArray[np.float64]

_K = 8


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("bootstrapped_dqn needs the torch `nn` extra") from exc


def bench_bootstrapped_dqn(
    seed: int = 29,
    n_train: int = 600,
    iters: int = 300,
    batch: int = 64,
    mc_eval: int = 2000,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s, x = synth_reward_env(n_train, rng)
    xt = torch.tensor(x).float()
    heads = [
        torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        for _ in range(_K)
    ]
    params = [p for hd in heads for p in hd.parameters()]
    opt = torch.optim.Adam(params, lr=3e-3)
    masks = rng.random((_K, n_train)) < 0.7
    for _it in range(iters):
        idx = rng.integers(0, n_train, batch)
        xb = xt[idx]
        a_take = rng.integers(0, 2, batch)
        rew = torch.tensor(sample_return(s[idx], a_take.astype(float), rng))
        loss = torch.zeros(())
        for k, hd in enumerate(heads):
            m = torch.tensor(masks[k, idx].astype(float)).float()
            q = hd(xb)[np.arange(batch), a_take]
            loss = loss + (m * (q - rew).pow(2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q_all = np.stack([hd(xt).numpy() for hd in heads], 0)
    q_mean = q_all.mean(0)
    q_lcb = q_all.min(0)
    per_state_lcb = np.zeros((4, 2))
    per_state_mean = np.zeros((4, 2))
    spread = np.zeros((4, 2))
    err = np.zeros((4, 2))
    eval_rng = np.random.default_rng(seed + 1)
    for st in range(4):
        mask = s == st
        per_state_lcb[st] = q_lcb[mask].mean(0)
        per_state_mean[st] = q_mean[mask].mean(0)
        spread[st] = q_all[:, mask].std(0).mean(0)
        for a in range(2):
            err[st, a] = abs(per_state_mean[st, a] - mc_return_dist(st, a, eval_rng, 800).mean())
    a_pess = per_state_lcb.argmax(1)
    a_mean = per_state_mean.argmax(1)
    cvar_pess = np.mean(
        [cvar(mc_return_dist(st, int(a_pess[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    cvar_mean = np.mean(
        [cvar(mc_return_dist(st, int(a_mean[st]), eval_rng, mc_eval)) for st in range(4)]
    )
    corr = float(np.corrcoef(spread.ravel(), err.ravel())[0, 1])
    return {
        "synthetic_boot_cvar": float(cvar_pess),
        "synthetic_boot_mean_cvar": float(cvar_mean),
        "synthetic_boot_cvar_gain": float(cvar_pess - cvar_mean),
        "synthetic_boot_spread_err_corr": corr if np.isfinite(corr) else 0.0,
        "synthetic_torch_available": 1.0,
    }
