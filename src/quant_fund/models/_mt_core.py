"""Shared two-task trainer for wave-178: shared-trunk MLP, per-task (SYNTHETIC)
heads; gradient combiners differ per method.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quant_fund.models._mt_synth import mt_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mt_core requires torch (pip install -e .[nn])") from exc
    return torch


def train_mtl(
    combine: Callable[[Any, Any], Any],
    seed: int,
    iters: int = 600,
    naive: bool = False,
) -> tuple[float, float]:
    """Returns (min_task_acc, mean_acc)."""
    torch = _torch()
    X, y1, y2 = mt_data(seed)
    Xt, yt1, yt2 = mt_data(seed + 1, n=200)
    torch.manual_seed(seed)
    D = 24
    trunk = torch.nn.Sequential(torch.nn.Linear(8, D), torch.nn.Tanh())
    h1 = torch.nn.Linear(D, 1)
    h2 = torch.nn.Linear(D, 1)
    params = list(trunk.parameters()) + list(h1.parameters()) + list(h2.parameters())
    opt = torch.optim.SGD(params, lr=0.05)
    Xt_ = torch.tensor(X).float()
    y1_ = torch.tensor(y1).float()
    y2_ = torch.tensor(y2).float()
    for _ in range(iters):
        h = trunk(Xt_)
        l1 = torch.nn.functional.binary_cross_entropy_with_logits(h1(h).squeeze(-1), y1_)
        l2 = torch.nn.functional.binary_cross_entropy_with_logits(h2(h).squeeze(-1), y2_)
        if naive:
            opt.zero_grad()
            (l1 + l2).backward()
        else:
            g1 = torch.autograd.grad(l1, trunk.parameters(), retain_graph=True)
            g2 = torch.autograd.grad(l2, trunk.parameters(), retain_graph=True)
            g = combine(
                torch.cat([x.reshape(-1) for x in g1]),
                torch.cat([x.reshape(-1) for x in g2]),
            )
            opt.zero_grad()
            (l1 + l2).backward()  # head grads; trunk grads overwritten below
            with torch.no_grad():
                off = 0
                for p in trunk.parameters():
                    n = p.numel()
                    p.grad = g[off : off + n].reshape(p.shape)
                    off += n
        opt.step()
    with torch.no_grad():
        h = trunk(torch.tensor(Xt).float())
        a1 = float(((h1(h).squeeze(-1) > 0).float().numpy() == yt1).mean())
        a2 = float(((h2(h).squeeze(-1) > 0).float().numpy() == yt2).mean())
    return min(a1, a2), (a1 + a2) / 2
