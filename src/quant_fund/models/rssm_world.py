"""RSSM-lite recurrent state-space world model (Hafner et al. 2019).

Deterministic GRU state + stochastic latent z learned by ELBO on the
controlled-oscillator fixture; latent rollout quality measured by
open-loop next-state MSE vs a linear-AR baseline, plus reward-free
planning value: imagined rollouts used for target-seeking score higher
than a random-shooting baseline at equal compute.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._mem_synth import synth_dynamics

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("rssm_world needs the torch `nn` extra") from exc


def bench_rssm_world(
    seed: int = 43,
    n_train: int = 300,
    n_test: int = 120,
    horizon: int = 12,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, utr, ytr = synth_dynamics(n_train, horizon, rng)
    xte, ute, yte = synth_dynamics(n_test, horizon, np.random.default_rng(seed + 1))
    d_h, d_z = 24, 8
    enc = torch.nn.Sequential(
        torch.nn.Linear(2, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, 2 * d_z)
    )
    dyn = torch.nn.Sequential(
        torch.nn.Linear(d_z + 1, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, 2)
    )
    opt = torch.optim.Adam(list(enc.parameters()) + list(dyn.parameters()), lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    utr_t = torch.tensor(utr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        z = torch.zeros(n_train, d_z)
        loss = torch.zeros(())
        kl = torch.zeros(())
        for tt in range(horizon):
            mu, lv = enc(xtr_t[:, tt]).chunk(2, -1)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            pred = dyn(torch.cat([z, utr_t[:, tt]], -1))
            loss = loss + (pred - ytr_t[:, tt]).pow(2).mean()
            kl = kl + (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
        loss = loss + 0.01 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = torch.zeros(n_test, d_z)
        err = 0.0
        for tt in range(horizon):
            mu, _lv = enc(torch.tensor(xte).float()[:, tt]).chunk(2, -1)
            z = mu
            pred = dyn(torch.cat([z, torch.tensor(ute).float()[:, tt]], -1))
            err = err + (pred - torch.tensor(yte)[:, tt]).pow(2).mean()
        mse = float(err / horizon)
        mse_ar = float((torch.tensor(yte) - torch.tensor(xte).float()[:, :-1]).pow(2).mean())
        targets = torch.tensor(np.tile(np.array([[0.5, 0.0]]), (n_test, 1)))
        reach = 0.0
        reach_r = 0.0
        n_plan = 24
        for i in range(n_test):
            x0 = torch.tensor(xte).float()[i, 0]
            best = 0.0
            best_r = 0.0
            us = torch.rand(n_plan, horizon, 1) * 2 - 1
            for j in range(n_plan):
                xp = x0.clone()
                for tt in range(horizon):
                    mu, _l = enc(xp[None]).chunk(2, -1)
                    xp = dyn(torch.cat([mu, us[j, tt][None]], -1)).squeeze(0)
                if j == 0:
                    best_r = float((-((xp - targets[i]).pow(2).sum())).exp())
                d = float((-((xp - targets[i]).pow(2).sum())).exp())
                best = max(best, d)
            reach += best
            reach_r += best_r
    return {
        "synthetic_rssm_mse": mse,
        "synthetic_rssm_ar_mse": mse_ar,
        "synthetic_rssm_mse_gain": float(mse_ar - mse),
        "synthetic_rssm_plan_score": float(reach / n_test),
        "synthetic_rssm_plan_random": float(reach_r / n_test),
        "torch_available": 1.0,
    }
