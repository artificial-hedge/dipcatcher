"""Tent test-time entropy minimization (Wang et al. 2021) (SYNTHETIC).

At test time only affine scale/shift parameters update by minimizing
prediction entropy on unlabeled shifted data — recovers accuracy lost to
the covariate shift without any labels.
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
        raise ImportError("tent_tta needs the torch `nn` extra") from exc


def bench_tent_tta(
    seed: int = 59,
    n_train: int = 400,
    n_test: int = 300,
    iters: int = 500,
    tta_iters: int = 150,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = synth_tta_split(n_train, n_test, rng)
    backbone = torch.nn.Sequential(
        torch.nn.Linear(16, 48), torch.nn.ReLU(), torch.nn.Linear(48, 32), torch.nn.ReLU()
    )
    affine = torch.nn.Parameter(torch.cat([torch.ones(1, 32), torch.zeros(1, 32)], 0))
    head = torch.nn.Linear(32, 4)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)
    for _i in range(iters):
        h = backbone(xtr_t)
        h = h * affine[0][None, :] + affine[1][None, :]
        loss = torch.nn.functional.cross_entropy(head(h), ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    xte_t = torch.tensor(xte).float()
    yte_t = torch.tensor(yte)
    with torch.no_grad():
        h = backbone(xte_t) * affine[0][None, :] + affine[1][None, :]
        acc_before = float((head(h).argmax(-1) == yte_t).float().mean())
    opt_t = torch.optim.Adam([affine], lr=1e-2)
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)
    for _i in range(tta_iters):
        h = backbone(xte_t)
        h = h * affine[0][None, :] + affine[1][None, :]
        logits = head(h)
        p = torch.softmax(logits, -1)
        loss = -(p * torch.log(p + 1e-9)).sum(-1).mean()
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
    with torch.no_grad():
        h = backbone(xte_t) * affine[0][None, :] + affine[1][None, :]
        acc_after = float((head(h).argmax(-1) == yte_t).float().mean())
    return {
        "synthetic_tent_acc_before": acc_before,
        "synthetic_tent_acc_after": acc_after,
        "synthetic_tent_gain": acc_after - acc_before,
        "synthetic_torch_available": 1.0,
    }
