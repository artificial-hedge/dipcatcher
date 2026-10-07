"""NAS fixture: a discrete architecture space over small MLPs (SYNTHETIC).

Arch = (hidden ∈ {4,8,16,24,48}, depth ∈ {1,2,3}, act ∈ {0=relu,1=tanh})
evaluated by short training on the compression synth split.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import acc_of

HIDDS = [4, 8, 16, 24, 48]
DEPTHS = [1, 2, 3]
ACTS = [0, 1]


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("NAS modules require torch (pip install -e .[nn])") from exc
    return torch


def build_net(torch, arch, seed: int = 0):
    torch.manual_seed(seed)
    h, d, a = arch
    act = torch.nn.ReLU() if a == 0 else torch.nn.Tanh()
    layers = [torch.nn.Linear(8, h), act]
    for _i in range(d - 1):
        layers += [torch.nn.Linear(h, h), act]
    layers.append(torch.nn.Linear(h, 2))
    return torch.nn.Sequential(*layers)


def eval_arch(
    arch, x_tr_t, y_tr_t, x_te_t, y_te_t, iters: int = 40, lr: float = 0.02, seed: int = 0
) -> float:
    torch = _torch()
    net = build_net(torch, arch, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_tr_t), y_tr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return acc_of(torch, net, x_te_t, y_te_t)


def noisy_labels(y, seed: int, frac: float = 0.2):
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    flips = rng.choice(len(y2), int(frac * len(y2)), replace=False)
    y2[flips] = 1 - y2[flips]
    return y2


def rand_arch(rng: np.random.Generator) -> tuple[int, int, int]:
    return (
        int(HIDDS[rng.integers(len(HIDDS))]),
        int(DEPTHS[rng.integers(len(DEPTHS))]),
        int(rng.integers(2)),
    )


def all_archs() -> list[tuple[int, int, int]]:
    return [(h, d, a) for h in HIDDS for d in DEPTHS for a in ACTS]
