"""Adafactor (Shazeer-Stern 2018) — factored second-moment estimate
for matrices: row/col RMS factors → O(n+m) memory vs Adam's O(nm).
"""

from __future__ import annotations

from quant_fund.models._opt_synth import opt_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("adafactor_opt requires torch (pip install -e .[nn])") from exc
    return torch


def bench_adafactor_opt(
    seed: int = 683,
    iters: int = 150,
) -> dict[str, float]:
    torch = _torch()
    x, y = opt_task(seed)
    xt = torch.tensor(x).float()
    yt = torch.tensor(y).float()

    def fwd(w1, b1, w2):
        return ((xt @ w1 + b1) @ w2).squeeze(-1)

    torch.manual_seed(seed)
    w1 = torch.randn(16, 8, requires_grad=True)
    b1 = torch.zeros(8, requires_grad=True)
    w2 = torch.randn(8, 1, requires_grad=True)
    # factored state only for w1 (the 2-D param); b1/w2 get 1-D rms
    r1 = torch.zeros(16)
    c1 = torch.zeros(8)
    v_b = torch.zeros(8)
    v2 = torch.zeros(8, 1)
    for t in range(1, iters + 1):
        loss = ((fwd(w1, b1, w2) - yt) ** 2).mean()
        loss.backward()
        decay = min(0.999, 1 - (t + 1) ** -0.8)
        with torch.no_grad():
            g1 = w1.grad
            r1 = decay * r1 + (1 - decay) * (g1 * g1).mean(1)
            c1 = decay * c1 + (1 - decay) * (g1 * g1).mean(0)
            vhat = (r1[:, None] * c1[None]) / c1.mean().clamp_min(1e-9)
            w1 -= 0.05 * g1 / (vhat.sqrt() + 1e-9)
            v_b = decay * v_b + (1 - decay) * b1.grad**2
            b1 -= 0.05 * b1.grad / (v_b.sqrt() + 1e-9)
            v2 = decay * v2 + (1 - decay) * w2.grad**2
            w2 -= 0.05 * w2.grad / (v2.sqrt() + 1e-9)
            for p in (w1, b1, w2):
                p.grad = None
    loss_af = float(((fwd(w1, b1, w2) - yt) ** 2).mean())
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
    # memory: Adafactor stores n+m for w1 vs Adam's 2nm
    mem_ratio = float((16 + 8) / (2 * 16 * 8))
    return {
        "synthetic_af_loss": loss_af,
        "synthetic_af_adam_loss": loss_a,
        "synthetic_af_gain": loss_a - loss_af,
        "synthetic_af_mem_ratio": mem_ratio,
        "synthetic_torch_available": 1.0,
    }
