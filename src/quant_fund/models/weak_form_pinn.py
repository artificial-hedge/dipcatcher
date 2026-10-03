"""Weak-form / variational PINN (Kharazmi et al. 2019) — test-function
weighted residual ∫r·v instead of pointwise residual; variance-reduced,
L2 error vs analytic.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._heat_synth import SIG, T_, L, eval_error, u0


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("weak_form_pinn requires torch (pip install -e .[nn])") from exc
    return torch


def bench_weak_form_pinn(seed: int = 2507, iters: int = 800) -> dict[str, float]:
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
        # weak residual: ∫ ut·v + 0.5σ² ∫ ux·v_x = 0, v = (1-|x|/L)_+ tent
        v = torch.clamp(1 - x.abs() / L, min=0.0).squeeze(1)
        vx = torch.where(x.squeeze(1) > 0, torch.tensor(-1.0 / L), torch.tensor(1.0 / L))
        loss = ((ut * v + 0.5 * SIG**2 * ux * vx).mean()) ** 2 * 1000
        x0 = torch.tensor(rng.uniform(-L, L, 256)).float().unsqueeze(1)
        t0 = torch.zeros(256, 1)
        ic = (
            (net(torch.cat([x0, t0], 1)).squeeze(1) - torch.tensor(u0(x0.numpy()[:, 0])).float())
            ** 2
        ).mean()
        loss = loss + 10 * ic
        opt.zero_grad()
        loss.backward()
        opt.step()

    def pred(xq: np.ndarray, tq: float) -> np.ndarray:
        with torch.no_grad():
            q = torch.cat([torch.tensor(xq).float().unsqueeze(1), torch.full((len(xq), 1), tq)], 1)
            return np.asarray(net(q).squeeze(1).numpy())

    return {"synthetic_weak_rel_l2": eval_error(pred), "torch_available": 1.0}
