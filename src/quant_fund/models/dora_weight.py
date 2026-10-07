"""DoRA weight-decomposed adaptation (Liu et al. 2024).

W' = m · (V + BA) / ||V + BA|| — direction adapted by low-rank BA,
magnitude m learned separately. On the rotated-boundary fixture it
matches LoRA's adaptation at the same rank while keeping the base's
row-norm geometry explicit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._peft_synth import synth_peft_base, synth_peft_shift

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("dora_weight needs the torch `nn` extra") from exc


def bench_dora_weight(
    seed: int = 139,
    n_train: int = 400,
    n_shift: int = 200,
    iters_base: int = 400,
    iters_adapt: int = 300,
    rank: int = 1,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_peft_base(n_train, rng)
    xs, ys = synth_peft_shift(n_shift, np.random.default_rng(seed + 1))
    base = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    x_t, y_t = torch.tensor(x).float(), torch.tensor(y)
    opt = torch.optim.Adam(base.parameters(), lr=5e-3)
    for _i in range(iters_base):
        loss = torch.nn.functional.cross_entropy(base(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in base.parameters():
        p.requires_grad_(False)
    m1 = torch.nn.Parameter(base[0].weight.norm(dim=1, keepdim=True).clone())
    m2 = torch.nn.Parameter(base[2].weight.norm(dim=1, keepdim=True).clone())
    a1 = torch.nn.Parameter(torch.randn(4, rank) * 0.1)
    r1 = torch.nn.Parameter(torch.zeros(rank, 32))
    a2 = torch.nn.Parameter(torch.randn(32, rank) * 0.1)
    r2 = torch.nn.Parameter(torch.zeros(rank, 2))
    xs_t, ys_t = torch.tensor(xs).float(), torch.tensor(ys)
    opt2 = torch.optim.Adam([m1, m2, a1, r1, a2, r2], lr=5e-3)

    def dora_w(w, m, delta):
        v = w + delta
        return m * v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def fwd(xb):
        w1 = dora_w(base[0].weight, m1, (a1 @ r1).T)
        w2 = dora_w(base[2].weight, m2, (a2 @ r2).T)
        return torch.relu(xb @ w1.T + base[0].bias) @ w2.T + base[2].bias

    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(fwd(xs_t), ys_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_d = float((fwd(xs_t).argmax(-1) == ys_t).float().mean())
        acc_d_t0 = float((fwd(x_t).argmax(-1) == y_t).float().mean())
    n_params = m1.numel() + m2.numel() + a1.numel() + r1.numel() + a2.numel() + r2.numel()
    n_full = sum(p.numel() for p in base.parameters())
    return {
        "synthetic_dora_acc_shift": acc_d,
        "synthetic_dora_t0_acc": acc_d_t0,
        "synthetic_dora_param_frac": float(n_params) / n_full,
        "synthetic_dora_m_shift": float(
            (m1 - base[0].weight.norm(dim=1, keepdim=True)).abs().mean()
        ),
        "synthetic_torch_available": 1.0,
    }
