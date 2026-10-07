"""Model-predictive shooting control over a learned dynamics model.

A dynamics model (trained on random actions) drives CEM-style shooting
planning toward a moving target. Metrics: final-state distance vs a
no-model random-action baseline and vs a proportional controller.
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
        raise ImportError("mpc_planning needs the torch `nn` extra") from exc


def bench_mpc_planning(
    seed: int = 47,
    n_train: int = 300,
    n_ep: int = 40,
    horizon: int = 10,
    plan_steps: int = 6,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, utr, ytr = synth_dynamics(n_train, horizon, rng)
    dyn = torch.nn.Sequential(torch.nn.Linear(3, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2))
    opt = torch.optim.Adam(dyn.parameters(), lr=3e-3)
    xtr_t = torch.tensor(xtr[:, :-1]).float().reshape(-1, 2)
    utr_t = torch.tensor(utr).float().reshape(-1, 1)
    ytr_t = torch.tensor(ytr).float().reshape(-1, 2)
    for _i in range(iters):
        idx = rng.integers(0, xtr_t.shape[0], 256)
        loss = (dyn(torch.cat([xtr_t[idx], utr_t[idx]], -1)) - ytr_t[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    dyn_err = float((dyn(torch.cat([xtr_t, utr_t], -1)) - ytr_t).pow(2).mean().detach())
    a_true = np.array([[0.9, -0.4], [0.4, 0.9]])
    b_true = np.array([0.3, 0.15])
    dist_model = 0.0
    dist_rand = 0.0
    dist_prop = 0.0
    plan_rng = np.random.default_rng(seed + 2)
    for _ep in range(n_ep):
        x = plan_rng.normal(0, 0.5, 2)
        xr = x.copy()
        xp = x.copy()
        target = np.array([0.5, 0.0])
        n_cand = 24
        for _s in range(plan_steps):
            cand = plan_rng.uniform(-1, 1, (n_cand, horizon - _s % horizon, 1))
            xs = torch.tensor(np.tile(x, (n_cand, 1))).float()
            for tt in range(cand.shape[1]):
                with torch.no_grad():
                    xs = dyn(torch.cat([xs, torch.tensor(cand[:, tt]).float()], -1))
            d = (xs - torch.tensor(target).float()).pow(2).sum(-1)
            u = float(cand[int(d.argmin()), 0, 0])
            x = x @ a_true.T + u * b_true
            xr = xr @ a_true.T + plan_rng.uniform(-1, 1) * b_true
            u_p = float(np.clip((target - xp) @ b_true / (b_true @ b_true), -1, 1))
            xp = xp @ a_true.T + u_p * b_true
        dist_model += float(np.linalg.norm(x - target))
        dist_rand += float(np.linalg.norm(xr - target))
        dist_prop += float(np.linalg.norm(xp - target))
    return {
        "synthetic_mpc_dyn_mse": dyn_err,
        "synthetic_mpc_dist": float(dist_model / n_ep),
        "synthetic_mpc_random_dist": float(dist_rand / n_ep),
        "synthetic_mpc_prop_dist": float(dist_prop / n_ep),
        "synthetic_mpc_gain": float((dist_rand - dist_model) / n_ep),
        "synthetic_torch_available": 1.0,
    }
