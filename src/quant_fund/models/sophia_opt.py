"""Sophia-lite (Liu et al. 2023) — diagonal-Hessian second-order update:
clipped m / (h + ε) with Hutchinson Hessian estimate — vs Adam.
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("sophia_opt requires torch (pip install -e .[nn])") from exc
    return torch


def bench_sophia_opt(
    seed: int = 673,
    iters: int = 100,
    hess_every: int = 5,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def fwd(w1, b1, w2):
        return ((xt @ w1 + b1) @ w2).squeeze(-1)

    torch.manual_seed(seed)
    params = [
        torch.randn(16, 8, requires_grad=True),
        torch.zeros(8, requires_grad=True),
        torch.randn(8, 1, requires_grad=True),
    ]
    w1, b1, w2 = params
    m = [torch.zeros_like(p) for p in params]
    h = [torch.zeros_like(p) for p in params]
    for i in range(iters):
        pred = fwd(w1, b1, w2)
        loss = ((pred - yt) ** 2).mean()
        loss.backward(create_graph=True)
        gs = torch.autograd.grad(loss, params, create_graph=i % hess_every == 0)
        with torch.no_grad():
            for j, p in enumerate(params):
                m[j] = 0.9 * m[j] + 0.1 * p.grad
        if i % hess_every == 0:
            # Hutchinson: h ≈ mean over probes of ∇(g·v)
            probe_h = [torch.zeros_like(p) for p in params]
            for _s in range(2):
                v = [torch.randint_like(p, 2).float() * 2 - 1 for p in params]
                gv = sum((g * vv).sum() for g, vv in zip(gs, v, strict=True))
                hv = torch.autograd.grad(gv, params, retain_graph=True)
                for j in range(len(params)):
                    probe_h[j] += v[j] * hv[j]
            for j in range(len(params)):
                h[j] = 0.99 * h[j] + 0.01 * probe_h[j].abs() * 10
        with torch.no_grad():
            for j, p in enumerate(params):
                upd = (m[j] / (h[j].clamp_min(1e-3) + 1e-3)).clamp(-1, 1)
                p -= 0.05 * upd
                p.grad = None
    loss_s = float(((fwd(*params) - yt) ** 2).mean())
    torch.manual_seed(seed)
    w1a = torch.randn(16, 8, requires_grad=True)
    b1a = torch.zeros(8, requires_grad=True)
    w2a = torch.randn(8, 1, requires_grad=True)
    opt = torch.optim.Adam([w1a, b1a, w2a], lr=0.01)
    for _i in range(iters):
        loss = ((fwd(w1a, b1a, w2a) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss_a = float(((fwd(w1a, b1a, w2a) - yt) ** 2).mean())
    return {
        "synthetic_sophia_loss": loss_s,
        "synthetic_sophia_adam_loss": loss_a,
        "synthetic_sophia_gain": loss_a - loss_s,
        "torch_available": 1.0,
    }
