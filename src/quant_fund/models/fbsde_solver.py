"""Deep BSDE solver (Han-Jentzen-E 2018) — heat equation via
forward-backward SDE: u(T,x) = E[u0(x + σ√T·Z)]; a network learns the
spatial gradient to walk u backward; L2 error vs analytic.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, L, eval_error, u0


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("fbsde_solver requires torch (pip install -e .[nn])") from exc
    return torch


def bench_fbsde_solver(seed: int = 2513, iters: int = 600, n_steps: int = 20) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    # learn u(t=0·, ·) approx then propagate; here learn u0(x)-param u_net(0,x)
    # plus gradient nets per step — simplify: single net u(t,x), BSDE loss
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    dt = T_ / n_steps
    for _ in range(iters):
        x = torch.zeros(256, 1).uniform_(-L, L)
        u_start = net(torch.cat([x, torch.zeros(256, 1)], 1))
        for k in range(n_steps):
            t = k * dt
            xt = torch.cat([x, torch.full((256, 1), t)], 1).requires_grad_(True)
            u = net(xt)
            ux = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 0:1]
            dw = np.sqrt(dt) * torch.randn(256, 1)
            x = x + SIG * dw
            u = u + 0.5 * SIG**2 * ux * dw  # Itô: du = ut dt + ux dW; u_t+0.5σ²u_xx=0 → du = ux dW
        terminal = torch.tensor(u0(x.detach().numpy()[:, 0])).float().unsqueeze(1)
        loss = ((u - terminal) ** 2).mean() + 0.01 * (
            (u_start - net(torch.cat([x * 0, torch.zeros(256, 1)], 1))) ** 2
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        with torch.no_grad():
            q = torch.cat([torch.tensor(xq).float().unsqueeze(1), torch.full((len(xq), 1), tq)], 1)
            return np.asarray(net(q).squeeze(1).numpy())

    return {"synthetic_bsde_rel_l2": eval_error(pred), "torch_available": 1.0}
