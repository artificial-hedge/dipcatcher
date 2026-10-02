"""Lion (Chen et al. 2023) — sign-momentum update:
w -= lr * sign(β1·m + (1-β1)·g), m ← β2·m + (1-β2)·g.
Memory-light vs Adam.
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lion_opt requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lion_opt(
    seed: int = 671,
    iters: int = 120,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def run(opt_fn, lr):
        torch.manual_seed(seed)
        w1 = torch.randn(16, 8, requires_grad=True)
        b1 = torch.zeros(8, requires_grad=True)
        w2 = torch.randn(8, 1, requires_grad=True)
        params = [w1, b1, w2]
        state = opt_fn(params)
        for _i in range(iters):
            pred = ((xt @ w1 + b1) @ w2).squeeze(-1)
            loss = ((pred - yt) ** 2).mean()
            loss.backward()
            state.step(params, lr)
            for p in params:
                p.grad = None
        return float(((((xt @ w1 + b1) @ w2).squeeze(-1) - yt) ** 2).mean())

    class _Lion:
        def __init__(self, params):
            self.m = [torch.zeros_like(p) for p in params]

        def step(self, params, lr, b1=0.9, b2=0.99):
            with torch.no_grad():
                for i, p in enumerate(params):
                    g = p.grad
                    upd = torch.sign(b1 * self.m[i] + (1 - b1) * g)
                    p -= lr * upd
                    self.m[i] = b2 * self.m[i] + (1 - b2) * g

    loss_l = run(lambda ps: _Lion(ps), 0.01)

    class _Adam:
        def __init__(self, params):
            self.opt = torch.optim.Adam(params, lr=0.01)

        def step(self, params, lr):
            self.opt.step()

    loss_a = run(lambda ps: _Adam(ps), 0.01)
    return {
        "synthetic_lion_loss": loss_l,
        "synthetic_lion_adam_loss": loss_a,
        "synthetic_lion_gain": loss_a - loss_l,
        "torch_available": 1.0,
    }
