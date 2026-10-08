"""Muon (Jordan et al. 2024) — momentum + Newton-Schulz orthogonalized (SYNTHETIC)
update for 2-D weights: the update is pulled toward a semi-orthogonal
matrix — vs Adam on the ill-conditioned quadratic task.
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("muon_opt requires torch (pip install -e .[nn])") from exc
    return torch


def _ns(G, steps: int = 5):
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G / (G.norm() + 1e-7)
    for _i in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


def bench_muon_opt(
    seed: int = 667,
    iters: int = 120,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def loss_of(w):
        pred = xt @ w
        return ((pred - yt) ** 2).mean()

    # Muon on a 16×16 weight reshaped from the linear map (d×1→4×4 impossible
    # for d=16 → use a two-layer net: 16→8→1, Muon on the 16×8 matrix)
    torch.manual_seed(seed)
    w1 = torch.randn(16, 8, requires_grad=True)
    b1 = torch.zeros(8, requires_grad=True)
    w2 = torch.randn(8, 1, requires_grad=True)
    mom = torch.zeros(16, 8)
    lr = 0.01
    for _i in range(iters):
        pred = (xt @ w1 + b1) @ w2
        pred = pred.squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            g1 = w1.grad
            mom = 0.9 * mom + g1
            w1 -= lr * _ns(mom)
            w2 -= lr * w2.grad
            b1 -= lr * b1.grad
            for p in (w1, w2, b1):
                p.grad = None
    loss_m = float(((((xt @ w1 + b1) @ w2).squeeze(-1) - yt) ** 2).mean())
    # Adam baseline
    torch.manual_seed(seed)
    w1a = torch.randn(16, 8, requires_grad=True)
    b1a = torch.zeros(8, requires_grad=True)
    w2a = torch.randn(8, 1, requires_grad=True)
    opt = torch.optim.Adam([w1a, b1a, w2a], lr=0.01)
    for _i in range(iters):
        pred = ((xt @ w1a + b1a) @ w2a).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss_a = float(((((xt @ w1a + b1a) @ w2a).squeeze(-1) - yt) ** 2).mean())
    return {
        "synthetic_muon_loss": loss_m,
        "synthetic_muon_adam_loss": loss_a,
        "synthetic_muon_gain": loss_a - loss_m,
        "synthetic_torch_available": 1.0,
    }
