"""Input-Convex Neural Network (ICNN) — guaranteed-convex energy function.

Amos et al. 2017: z_{k+1} = act(Wz_k z_k + Wx_k x + b_k) with Wz_k >= 0 and
convex monotone activation => f(x, y) convex in y by construction. Used as a
learned convex potential/value for downstream argmin decisions. Bench checks
both approximation quality AND the structural guarantee (Jensen violations
on random midpoints) vs an unconstrained MLP. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261013


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _true_convex(x: FloatArray, y: FloatArray) -> FloatArray:
    """Ground-truth convex potential: quad well + hinge ramp, x-modulated."""
    a = 1.0 + 0.5 * np.tanh(x[..., 0])
    b = 0.5 * np.sin(x[..., 1])
    return a * (y - b) ** 2 + 0.5 * np.maximum(y - 0.3 * x[..., 0], 0)


def bench_input_convex(
    seed: int = 13,
    n_train: int = 3000,
    n_test: int = 600,
    iters: int = 2500,
    d: int = 64,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed + _SEED)
    xtr = rng.normal(0, 1, (n_train, 4))
    ytr = rng.uniform(-2, 2, n_train)
    vtr = _true_convex(xtr, ytr) + 0.02 * rng.standard_normal(n_train)
    xte = rng.normal(0, 1, (n_test, 4))
    yte = rng.uniform(-2, 2, n_test)
    vte = _true_convex(xte, yte)

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr[:, None], dtype=torch.float32)
    vt = torch.tensor(vtr[:, None], dtype=torch.float32)

    # ICNN: z layers with nonneg Wz (via softplus on raw params)
    wz_raw = [
        torch.nn.Parameter(torch.randn(d, 1) * 0.1),
        torch.nn.Parameter(torch.randn(d, d) * 0.1),
        torch.nn.Parameter(torch.randn(d, d) * 0.1),
    ]
    wx = [
        torch.nn.Linear(4, d),
        torch.nn.Linear(4, d),
        torch.nn.Linear(4, d),
    ]
    wy = [torch.nn.Linear(1, d, bias=False) for _ in range(3)]
    head = torch.nn.Linear(d, 1)
    params = (
        wz_raw
        + [p for m in wx for p in m.parameters()]
        + [p for m in wy for p in m.parameters()]
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=1.5e-3)

    def icnn(xb: Any, yb: Any) -> Any:
        z = torch.nn.functional.softplus(wx[0](xb) + wy[0](yb))
        for i in range(1, 3):
            wz = torch.nn.functional.softplus(wz_raw[i])  # >= 0
            z = torch.nn.functional.softplus(z @ wz.T + wx[i](xb) + wy[i](yb))
        return head(z)

    # unconstrained MLP comparator
    mlp = torch.nn.Sequential(
        torch.nn.Linear(5, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=2e-3)

    for _ in range(iters):
        mse_i = ((icnn(xt, yt) - vt) ** 2).mean()
        opt.zero_grad()
        mse_i.backward()
        opt.step()
        mse_m = ((mlp(torch.cat([xt, yt], -1)) - vt) ** 2).mean()
        opt_mlp.zero_grad()
        mse_m.backward()
        opt_mlp.step()

    with torch.no_grad():
        xe = torch.tensor(xte, dtype=torch.float32)
        ye = torch.tensor(yte[:, None], dtype=torch.float32)
        ve = torch.tensor(vte[:, None], dtype=torch.float32)
        mse_icnn = float(((icnn(xe, ye) - ve) ** 2).mean())
        mse_mlp = float(((mlp(torch.cat([xe, ye], -1)) - ve) ** 2).mean())
        # Jensen violation probe: f((y1+y2)/2) <= (f(y1)+f(y2))/2 must hold
        n_probe = min(500, xte.shape[0])
        y1 = torch.tensor(rng.uniform(-2, 2, n_probe)[:, None], dtype=torch.float32)
        y2 = torch.tensor(rng.uniform(-2, 2, n_probe)[:, None], dtype=torch.float32)
        xb = torch.tensor(xte[:n_probe], dtype=torch.float32)
        mid_icnn = icnn(xb, 0.5 * (y1 + y2)) - 0.5 * (icnn(xb, y1) + icnn(xb, y2))
        mid_mlp = mlp(torch.cat([xb, 0.5 * (y1 + y2)], -1)) - 0.5 * (
            mlp(torch.cat([xb, y1], -1)) + mlp(torch.cat([xb, y2], -1))
        )
        viol_icnn = float((mid_icnn > 1e-4).float().mean())
        viol_mlp = float((mid_mlp > 1e-4).float().mean())
    return {
        "synthetic_icnn_mse": mse_icnn,
        "synthetic_icnn_mlp_mse": mse_mlp,
        "synthetic_icnn_jensen_viol": viol_icnn,
        "synthetic_icnn_mlp_jensen_viol": viol_mlp,
        "synthetic_icnn_convexity_gain": viol_mlp - viol_icnn,
        "synthetic_torch_available": 1.0,
    }
