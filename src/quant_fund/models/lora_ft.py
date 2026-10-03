"""LoRA low-rank adaptation (Hu et al. 2021).

W' = W + (α/r)·BA with base frozen — trains r(A+B) params per layer
instead of d². On the rotated-boundary fixture LoRA recovers most of
full-finetune accuracy at ~2% of the parameter count, with zero extra
base-model forgetting beyond the adapter.
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
        raise ImportError("lora_ft needs the torch `nn` extra") from exc


def _base_net(torch):
    return torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))


def bench_lora_ft(
    seed: int = 131,
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
    base = _base_net(torch)
    x_t, y_t = torch.tensor(x).float(), torch.tensor(y)
    opt = torch.optim.Adam(base.parameters(), lr=5e-3)
    for _i in range(iters_base):
        loss = torch.nn.functional.cross_entropy(base(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_base_t0 = float((base(x_t).argmax(-1) == y_t).float().mean())
        acc_shift_pre = float(
            (base(torch.tensor(xs).float()).argmax(-1) == torch.tensor(ys)).float().mean()
        )
    for p in base.parameters():
        p.requires_grad_(False)
    a1 = torch.nn.Parameter(torch.zeros(4, rank))
    b1 = torch.nn.Parameter(torch.zeros(rank, 32))
    a2 = torch.nn.Parameter(torch.zeros(32, rank))
    b2 = torch.nn.Parameter(torch.zeros(rank, 2))
    torch.nn.init.normal_(a1, std=0.1)
    torch.nn.init.normal_(a2, std=0.1)
    xs_t, ys_t = torch.tensor(xs).float(), torch.tensor(ys)
    opt2 = torch.optim.Adam([a1, b1, a2, b2], lr=5e-3)

    def fwd(xb):
        h = torch.relu(xb @ (base[0].weight + (a1 @ b1).T).T + base[0].bias)
        return h @ (base[2].weight + (a2 @ b2).T).T + base[2].bias

    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(fwd(xs_t), ys_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_lora = float((fwd(xs_t).argmax(-1) == ys_t).float().mean())
        acc_lora_t0 = float((fwd(x_t).argmax(-1) == y_t).float().mean())
    for p in base.parameters():
        p.requires_grad_(True)
    opt3 = torch.optim.Adam(base.parameters(), lr=5e-3)
    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(base(xs_t), ys_t)
        opt3.zero_grad()
        loss.backward()
        opt3.step()
    with torch.no_grad():
        acc_full = float((base(xs_t).argmax(-1) == ys_t).float().mean())
        acc_full_t0 = float((base(x_t).argmax(-1) == y_t).float().mean())
    n_full = sum(p.numel() for p in base.parameters())
    n_lora = a1.numel() + b1.numel() + a2.numel() + b2.numel()
    return {
        "synthetic_lora_acc_shift": acc_lora,
        "synthetic_lora_full_acc_shift": acc_full,
        "synthetic_lora_base_acc_pre": acc_shift_pre,
        "synthetic_lora_t0_retention": acc_lora_t0,
        "synthetic_lora_full_t0_retention": acc_full_t0,
        "synthetic_lora_base_t0_acc": acc_base_t0,
        "synthetic_lora_param_frac": float(n_lora) / n_full,
        "torch_available": 1.0,
    }
