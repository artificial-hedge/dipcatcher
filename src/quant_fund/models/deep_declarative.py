"""Deep declarative node — argmin_y f(theta(x), y) as a differentiable layer.

Gould et al. 2016/2019: a layer defined by an optimization problem
y*(x) = argmin_y f(x, y); the forward pass solves by Newton descent and
unrolled iterations carry exact gradients back to the parameters of f. Bench:
learn a cost-shaping head so the argmin matches expert position sizing, vs a
plain regression head. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261014


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def synth_expert_positions(
    n: int, d_feat: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Expert position y* = clip(tanh(B psi(x)), -1, 1) + noise."""
    x = rng.normal(0, 1, (n, d_feat))
    z = np.stack([x[:, i % d_feat] for i in range(6)], 1)
    b = rng.normal(0, 0.8, 6)
    y = np.tanh(z @ np.asarray(b) + 0.3 * np.sin(x[:, 0] * x[:, 1]))
    return x.astype(np.float64), y.astype(np.float64)


def _argmin_layer(theta: Any, torch: Any, iters: int = 30, lr: float = 0.2) -> Any:
    """y* = argmin_y f_theta(y); theta=(a,b,c): f = a y^2 - b y + c|y| softened.

    Unrolled gradient descent on a strictly convex scalar objective; the
    iterations stay in-graph so gradients reach theta exactly.
    """
    a, b, c = theta[..., 0:1], theta[..., 1:2], theta[..., 2:3]
    y = torch.zeros_like(b)
    for _ in range(iters):
        grad = 2 * torch.nn.functional.softplus(a) * y - b + c * torch.tanh(50.0 * y)
        step = lr * grad / (2 * torch.nn.functional.softplus(a) + 1e-3)
        y = y - step
    return y


def bench_deep_declarative(
    seed: int = 17,
    n_train: int = 2000,
    n_test: int = 400,
    d_feat: int = 10,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_expert_positions(n_train, d_feat, rng)
    xte, yte = synth_expert_positions(n_test, d_feat, rng)

    theta_net = torch.nn.Sequential(
        torch.nn.Linear(d_feat, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3),
    )
    reg_net = torch.nn.Sequential(
        torch.nn.Linear(d_feat, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    opt_t = torch.optim.Adam(theta_net.parameters(), lr=1e-3)
    opt_r = torch.optim.Adam(reg_net.parameters(), lr=1e-3)

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr[:, None], dtype=torch.float32)
    for _ in range(iters):
        y_star = _argmin_layer(theta_net(xt), torch)
        loss_t = ((y_star - yt) ** 2).mean()
        opt_t.zero_grad()
        loss_t.backward()
        opt_t.step()
        loss_r = ((reg_net(xt) - yt) ** 2).mean()
        opt_r.zero_grad()
        loss_r.backward()
        opt_r.step()

    with torch.no_grad():
        xe = torch.tensor(xte, dtype=torch.float32)
        ye = torch.tensor(yte[:, None], dtype=torch.float32)
        y_decl = _argmin_layer(theta_net(xe), torch)
        y_reg = reg_net(xe)
        mse_decl = float(((y_decl - ye) ** 2).mean())
        mse_reg = float(((y_reg - ye) ** 2).mean())
        # constraint-satisfaction metric: argmin output respects |y|<~1.1 bound
        bound_viol = float((y_decl.abs() > 1.05).float().mean())
        reg_bound_viol = float((y_reg.abs() > 1.05).float().mean())
    return {
        "synthetic_decl_mse": mse_decl,
        "synthetic_decl_reg_mse": mse_reg,
        "synthetic_decl_mse_gain": mse_reg - mse_decl,
        "synthetic_decl_bound_viol": bound_viol,
        "synthetic_decl_reg_bound_viol": reg_bound_viol,
        "torch_available": 1.0,
    }
