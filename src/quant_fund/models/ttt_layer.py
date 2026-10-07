"""Test-time training layer (Sun et al. 2020) (SYNTHETIC).

A self-supervised auxiliary rotation task trains a head jointly with the
main classifier; at test time the model re-adapts to each unlabeled batch
by minimizing the rotation-loss before predicting — recovers shift-lost
accuracy where the rotation task still carries signal.
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
        raise ImportError("ttt_layer needs the torch `nn` extra") from exc


def bench_ttt_layer(
    seed: int = 63,
    n_train: int = 400,
    n_test: int = 300,
    iters: int = 500,
    ttt_iters: int = 80,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = synth_tta_split(n_train, n_test, rng)
    shared = torch.nn.Sequential(
        torch.nn.Linear(16, 48), torch.nn.ReLU(), torch.nn.Linear(48, 32), torch.nn.ReLU()
    )
    cls_head = torch.nn.Linear(32, 4)
    rot_head = torch.nn.Linear(32, 4)
    params = list(shared.parameters()) + list(cls_head.parameters()) + list(rot_head.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xtr_t = torch.tensor(xtr).float()
    ytr_t = torch.tensor(ytr)

    def four_rot(xb):
        b = xb.shape[0]
        d = xb.shape[1] // 4
        outs = [xb]
        for _k in range(3):
            xb = torch.roll(xb, d, dims=1)
            outs.append(xb)
        x4 = torch.cat(outs)
        y4 = torch.arange(4).repeat_interleave(b)
        return x4, y4

    for _i in range(iters):
        x4, y4 = four_rot(xtr_t)
        h4 = shared(x4)
        h = shared(xtr_t)
        loss = torch.nn.functional.cross_entropy(
            cls_head(h), ytr_t
        ) + torch.nn.functional.cross_entropy(rot_head(h4), y4)
        opt.zero_grad()
        loss.backward()
        opt.step()
    xte_t = torch.tensor(xte).float()
    yte_t = torch.tensor(yte)
    with torch.no_grad():
        acc_before = float((cls_head(shared(xte_t)).argmax(-1) == yte_t).float().mean())
    for p in cls_head.parameters():
        p.requires_grad_(False)
    opt_t = torch.optim.Adam(list(shared.parameters()) + list(rot_head.parameters()), lr=5e-4)
    for _i in range(ttt_iters):
        x4, y4 = four_rot(xte_t)
        loss = torch.nn.functional.cross_entropy(rot_head(shared(x4)), y4)
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
    with torch.no_grad():
        acc_after = float((cls_head(shared(xte_t)).argmax(-1) == yte_t).float().mean())
    return {
        "synthetic_ttt_acc_before": acc_before,
        "synthetic_ttt_acc_after": acc_after,
        "synthetic_ttt_gain": acc_after - acc_before,
        "synthetic_torch_available": 1.0,
    }
