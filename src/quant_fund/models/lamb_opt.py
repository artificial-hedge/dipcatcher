"""LAMB (You et al. 2020) — Adam with layer-wise trust ratio:
update scaled by ||w||/||adam_step|| — vs Adam at matched steps.
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lamb_opt requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lamb_opt(
    seed: int = 679,
    iters: int = 120,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def fwd(w1, b1, w2):
        return ((xt @ w1 + b1) @ w2).squeeze(-1)

    class _Lamb:
        def __init__(self, params):
            self.m = [torch.zeros_like(p) for p in params]
            self.v = [torch.zeros_like(p) for p in params]
            self.t = 0

        def step(self, params, lr):
            self.t += 1
            with torch.no_grad():
                for i, p in enumerate(params):
                    g = p.grad
                    self.m[i] = 0.9 * self.m[i] + 0.1 * g
                    self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                    mh = self.m[i] / (1 - 0.9**self.t)
                    vh = self.v[i] / (1 - 0.999**self.t)
                    upd = mh / (vh.sqrt() + 1e-8)
                    r = p.norm() / (upd.norm() + 1e-9)
                    p -= lr * r.clamp(0, 10) * upd

    torch.manual_seed(seed)
    params = [
        torch.randn(16, 8, requires_grad=True),
        torch.zeros(8, requires_grad=True),
        torch.randn(8, 1, requires_grad=True),
    ]
    lamb = _Lamb(params)
    for _i in range(iters):
        loss = ((fwd(*params) - yt) ** 2).mean()
        loss.backward()
        lamb.step(params, lr=0.02)
        for p in params:
            p.grad = None
    loss_l = float(((fwd(*params) - yt) ** 2).mean())
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
        "synthetic_lamb_loss": loss_l,
        "synthetic_lamb_adam_loss": loss_a,
        "synthetic_lamb_gain": loss_a - loss_l,
        "torch_available": 1.0,
    }
