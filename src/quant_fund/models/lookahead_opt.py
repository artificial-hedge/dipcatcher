"""Lookahead (Zhang et al. 2019) — k fast steps then interpolate toward
the slow weights: φ ← φ + α(θ_k − φ) — vs plain Adam/SGD inner loop.
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lookahead_opt requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lookahead_opt(
    seed: int = 677,
    iters: int = 120,
    k: int = 6,
    alpha: float = 0.5,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def fwd(w1, b1, w2):
        return ((xt @ w1 + b1) @ w2).squeeze(-1)

    torch.manual_seed(seed)
    fast = [
        torch.randn(16, 8, requires_grad=True),
        torch.zeros(8, requires_grad=True),
        torch.randn(8, 1, requires_grad=True),
    ]
    slow = [p.detach().clone() for p in fast]
    opt = torch.optim.Adam(fast, lr=0.01)
    for i in range(iters):
        loss = ((fwd(*fast) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % k == 0:
            with torch.no_grad():
                for f, s in zip(fast, slow, strict=True):
                    s.add_(alpha * (f - s))
                    f.copy_(s)
    loss_look = float(((fwd(*fast) - yt) ** 2).mean())
    torch.manual_seed(seed)
    params = [
        torch.randn(16, 8, requires_grad=True),
        torch.zeros(8, requires_grad=True),
        torch.randn(8, 1, requires_grad=True),
    ]
    opt = torch.optim.Adam(params, lr=0.01)
    for _i in range(iters):
        loss = ((fwd(*params) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss_a = float(((fwd(*params) - yt) ** 2).mean())
    return {
        "synthetic_look_loss": loss_look,
        "synthetic_look_adam_loss": loss_a,
        "synthetic_look_gain": loss_a - loss_look,
        "synthetic_torch_available": 1.0,
    }
