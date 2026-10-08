"""DeepRitz PINN (E & Yu 2018) — energy-functional residual network (SYNTHETIC)
for the heat equation; L2 error vs analytic + FD-style baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, L, eval_error, u0


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deepritz_pinn requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deepritz_pinn(seed: int = 2501, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        x = torch.tensor(rng.uniform(-L, L, 256)).float().unsqueeze(1)
        t = torch.tensor(rng.uniform(0, T_, 256)).float().unsqueeze(1)
        xt = torch.cat([x, t], 1).requires_grad_(True)
        u = net(xt)
        g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        ut, ux = g[:, 1], g[:, 0]
        uxx = torch.autograd.grad(ux.sum(), xt, create_graph=True)[0][:, 0]
        pde = ((ut - 0.5 * SIG**2 * uxx) ** 2).mean()
        x0 = torch.tensor(rng.uniform(-L, L, 256)).float().unsqueeze(1)
        t0 = torch.zeros(256, 1)
        ic = (
            (net(torch.cat([x0, t0], 1)).squeeze(1) - torch.tensor(u0(x0.numpy()[:, 0])).float())
            ** 2
        ).mean()
        xb = torch.tensor([-L, L]).float().unsqueeze(1)
        tb = torch.tensor(rng.uniform(0, T_, 2)).float().unsqueeze(1)
        bc = (net(torch.cat([xb, tb], 1)) ** 2).mean()
        loss = pde + 10 * ic + 10 * bc
        opt.zero_grad()
        loss.backward()
        opt.step()

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        with torch.no_grad():
            q = torch.cat([torch.tensor(xq).float().unsqueeze(1), torch.full((len(xq), 1), tq)], 1)
            return np.asarray(net(q).squeeze(1).numpy())

    err = eval_error(pred)
    return {"synthetic_pinn_rel_l2": err, "synthetic_torch_available": 1.0}
