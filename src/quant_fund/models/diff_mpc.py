"""Differentiable MPC — learn a terminal cost so closed-loop MPC imitates.

Amos et al. 2018: an MPC controller (quadratic stage cost, linear dynamics,
finite horizon) is a differentiable policy; the terminal cost matrix P can be
learned by matching expert trajectories through unrolled LQR solves. Bench:
imitation loss vs behavior cloning on a position-tracking task. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261016


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _lqr_policy(a: Any, b: Any, q: Any, r: Any, p_term: Any, horizon: int) -> Any:
    """Discrete-time Riccati recursion -> time-varying feedback gains K_t."""
    p = p_term
    ks = []
    for _ in range(horizon):
        k = (r + b.T @ p @ b).inverse() @ (b.T @ p @ a)
        ks.append(k)
        p = q + a.T @ p @ (a - b @ k)
    return ks[::-1]


def bench_diff_mpc(
    seed: int = 25,
    n_train: int = 60,
    n_test: int = 20,
    iters: int = 300,
    horizon: int = 8,
    t_final: int = 30,
) -> dict[str, float]:
    torch = _torch()
    # expert: linear system x' = a x + b u, expert tracks sin target
    a = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    b = torch.tensor([[0.0], [0.3]])
    q = torch.diag(torch.tensor([1.0, 0.1]))
    r = torch.tensor([[0.05]])

    def expert_traj(seed_i: int) -> Any:
        rgx = np.random.default_rng(seed_i)
        x0 = torch.tensor(rgx.normal(0, 1, 2), dtype=torch.float32)
        target = torch.tensor(
            np.sin(np.linspace(0, 3, t_final) + rgx.uniform(0, 6)),
            dtype=torch.float32,
        )
        # expert gains: known good P (diag high position weight)
        p_star = torch.diag(torch.tensor([8.0, 1.0]))
        ks = _lqr_policy(a, b, q, r, p_star, horizon)
        x = x0
        traj = [x.clone()]
        for t in range(t_final):
            xt_ref = torch.tensor([target[t], 0.0])
            u = -ks[t % horizon] @ (x - xt_ref)
            x = a @ x + b[..., 0] * u[0]
            traj.append(x.clone())
        return x0, target, torch.stack(traj)

    trajs = [expert_traj(seed + i) for i in range(n_train)]
    test_trajs = [expert_traj(seed + 1000 + i) for i in range(n_test)]

    # learnable terminal cost P = L L^T + eps I (SPD)
    l_param = torch.nn.Parameter(torch.eye(2) * 0.5)
    opt = torch.optim.Adam([l_param], lr=3e-2)

    def mpc_traj(x0: Any, target: Any, p_term: Any) -> Any:
        ks = _lqr_policy(a, b, q, r, p_term, horizon)
        x = x0
        traj = [x]
        for t in range(t_final):
            xt_ref = torch.stack([target[t], torch.tensor(0.0)])
            u = -ks[t % horizon] @ (x - xt_ref)
            x = a @ x + b[..., 0] * u[0]
            traj.append(x)
        return torch.stack(traj)

    for _ in range(iters):
        p_term = l_param @ l_param.T + 1e-3 * torch.eye(2)
        loss = torch.zeros(())
        for x0, target, x_star in trajs[:20]:
            x_hat = mpc_traj(x0, target, p_term)
            loss = loss + ((x_hat - x_star) ** 2).mean()
        loss = loss / 20
        opt.zero_grad()
        loss.backward()
        opt.step()

    # behavior cloning baseline: map (x, target_t) -> u directly
    bc = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt_bc = torch.optim.Adam(bc.parameters(), lr=1e-2)
    feats, acts = [], []
    p_star = torch.diag(torch.tensor([8.0, 1.0]))
    ks = _lqr_policy(a, b, q, r, p_star, horizon)
    for x0, target, _x_star in trajs:
        x = x0
        for t in range(t_final):
            xt_ref = torch.stack([target[t], torch.tensor(0.0)])
            u = -ks[t % horizon] @ (x - xt_ref)
            feats.append(torch.cat([x, target[t].reshape(1)]))
            acts.append(u.reshape(1))
            x = a @ x + b[..., 0] * u[0]
    f_all = torch.stack(feats)
    u_all = torch.stack(acts)
    for _ in range(iters):
        loss_bc = ((bc(f_all) - u_all) ** 2).mean()
        opt_bc.zero_grad()
        loss_bc.backward()
        opt_bc.step()

    def closed_loop_err(policy: Any) -> float:
        errs = []
        for x0, target, x_star in test_trajs:
            x = x0
            err = 0.0
            for t in range(t_final):
                u = policy(x, target, t)
                x = a @ x + b[..., 0] * u[0]
                err += float((x - x_star[t + 1]).pow(2).mean())
            errs.append(err / t_final)
        return float(np.mean(errs))

    with torch.no_grad():
        p_learned = l_param @ l_param.T + 1e-3 * torch.eye(2)
        ks_l = _lqr_policy(a, b, q, r, p_learned, horizon)

        def mpc_pol(x: Any, target: Any, t: int) -> Any:
            xt_ref = torch.stack([target[t], torch.tensor(0.0)])
            return -ks_l[t % horizon] @ (x - xt_ref)

        def bc_pol(x: Any, target: Any, t: int) -> Any:
            _ = t
            return bc(torch.cat([x, target[t].reshape(1)])).reshape(-1)

        err_mpc = closed_loop_err(mpc_pol)
        err_bc = closed_loop_err(bc_pol)
    return {
        "synthetic_mpc_cl_err": err_mpc,
        "synthetic_mpc_bc_err": err_bc,
        "synthetic_mpc_err_gain": err_bc - err_mpc,
        "synthetic_torch_available": 1.0,
    }
