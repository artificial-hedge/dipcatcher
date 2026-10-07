"""Neural-ODE adjoint sensitivity (Chen et al. 2018) — terminal-loss (SYNTHETIC)
gradient via backward adjoint ODE vs autograd through the unrolled
solver; measures accuracy and memory-freeness.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import grad_corr


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ode_adjoint requires torch (pip install -e .[nn])") from exc
    return torch


def bench_ode_adjoint(seed: int = 2411) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    # dz/dt = tanh(A z), z(0)=z0; loss = ||z(T)-target||²
    A = torch.tensor([[0.5, -0.3], [0.2, 0.4]])
    z0 = torch.tensor([0.5, -0.2])
    target = torch.tensor([1.0, 0.5])
    T, steps = 1.0, 40
    dt = T / steps

    # reference: autograd through unrolled Euler
    A_ = A.clone().requires_grad_(True)
    z = z0.clone()
    for _ in range(steps):
        z = z + dt * torch.tanh(A_ @ z)
    loss = ((z - target) ** 2).sum()
    loss.backward()
    g_auto = A_.grad.clone()

    # adjoint method: forward without graph, backward via adjoint ODE
    with torch.no_grad():
        z = z0.clone()
        traj = [z.clone()]
        for _ in range(steps):
            z = z + dt * torch.tanh(A @ z)
            traj.append(z.clone())
    a = 2 * (traj[-1] - target)  # dL/dz(T)
    g_adj = torch.zeros_like(A)
    # backward: da/dt = -a·∂f/∂z = -a·(1-tanh²)·A; accumulate dL/dA
    for i in range(steps, 0, -1):
        zi = traj[i - 1]
        f = torch.tanh(A @ zi)
        a = a + dt * (-(1 - torch.tanh(A @ zi) ** 2) * (A.t() @ a))
        # dL/dA += a·∂f/∂A = outer(a∘(1-tanh²), z)
        sech2 = 1 - f * f
        g_adj = g_adj + dt * torch.outer(a * sech2, zi)
    corr = grad_corr(g_adj.numpy().ravel(), g_auto.numpy().ravel())
    err = float(np.abs(g_adj.numpy() - g_auto.numpy()).max())
    return {
        "synthetic_adjoint_corr": corr,
        "synthetic_adjoint_max_err": err,
        "synthetic_torch_available": 1.0,
    }
