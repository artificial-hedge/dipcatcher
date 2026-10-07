"""SHOT source-hypothesis transfer (Liang et al. 2020).

Freeze the classifier head; adapt the backbone on unlabeled shifted data
with information maximization (entropy + diversity terms) plus a
pseudo-label cross-entropy on confident samples.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._ssl_synth import synth_tta_split

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("shot_tta needs the torch `nn` extra") from exc


def bench_shot_tta(
    seed: int = 61,
    n_train: int = 400,
    n_test: int = 300,
    iters: int = 500,
    tta_iters: int = 200,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = synth_tta_split(n_train, n_test, rng)
    backbone = torch.nn.Sequential(
        torch.nn.Linear(16, 48), torch.nn.ReLU(), torch.nn.Linear(48, 32), torch.nn.ReLU()
    )
    head = torch.nn.Linear(32, 4)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(head(backbone(xtr_t)), ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    xte_t = torch.tensor(xte).float()
    yte_t = torch.tensor(yte)
    with torch.no_grad():
        acc_before = float((head(backbone(xte_t)).argmax(-1) == yte_t).float().mean())
    for p in head.parameters():
        p.requires_grad_(False)
    opt_t = torch.optim.Adam(backbone.parameters(), lr=2e-4)
    for _i in range(tta_iters):
        logits = head(backbone(xte_t))
        p = torch.softmax(logits, -1)
        ent = -(p * torch.log(p + 1e-9)).sum(-1).mean()
        div = (p.mean(0) * torch.log(p.mean(0) + 1e-9)).sum()
        conf = p.max(-1).values > 0.8
        pl_loss = torch.zeros(())
        if conf.any():
            pl_loss = torch.nn.functional.cross_entropy(logits[conf], p[conf].argmax(-1))
        loss = ent + div + 0.1 * pl_loss
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
    with torch.no_grad():
        acc_after = float((head(backbone(xte_t)).argmax(-1) == yte_t).float().mean())
    return {
        "synthetic_shot_acc_before": acc_before,
        "synthetic_shot_acc_after": acc_after,
        "synthetic_shot_gain": acc_after - acc_before,
        "synthetic_torch_available": 1.0,
    }
