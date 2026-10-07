"""DARTS — differentiable architecture search (Liu et al. 2019) (SYNTHETIC).

Continuous mixing weights α over candidate widths on a single-layer
net; bilevel alternation (weights on train, α on held-out slice);
discretize → final arch accuracy vs random pick.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import split
from quant_fund.models._nas_synth import _torch, eval_arch, noisy_labels


def bench_darts_nas(
    seed: int = 467,
    n: int = 300,
    iters: int = 60,
    hidds=(4, 8, 16, 24),
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    n_h = len(hidds)
    half = len(x_tr_t) // 2
    xw, yw = x_tr_t[:half], y_tr_t[:half]
    xa, ya = x_tr_t[half:], y_tr_t[half:]
    # supernet: one layer per candidate width, mixed by softmax(α)
    torch.manual_seed(seed)
    cells = [
        torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 2))
        for h in hidds
    ]
    alpha = torch.nn.Parameter(torch.zeros(n_h))
    params = [p for c in cells for p in c.parameters()]
    opt_w = torch.optim.Adam(params, lr=0.02)
    opt_a = torch.optim.Adam([alpha], lr=0.05)
    for _i in range(iters):
        # step weights on xw
        wts = torch.softmax(alpha, 0)
        out = sum(wts[k] * cells[k](xw) for k in range(n_h))
        loss = torch.nn.functional.cross_entropy(out, yw)
        opt_w.zero_grad()
        loss.backward()
        opt_w.step()
        # step alpha on xa
        wts = torch.softmax(alpha, 0)
        out_a = sum(wts[k] * cells[k](xa) for k in range(n_h))
        loss_a = torch.nn.functional.cross_entropy(out_a, ya)
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()
    pick = int(alpha.detach().argmax())
    # final: retrain discrete winner vs random pick
    arch = (hidds[pick], 1, 0)
    acc_pick = eval_arch(arch, x_tr_t, y_tr_t, x_te_t, y_te_t, 40, seed=7)
    acc_rand = float(
        np.mean(
            [
                eval_arch((h, 1, 0), x_tr_t, y_tr_t, x_te_t, y_te_t, 40, seed=k)
                for k, h in enumerate(hidds)
            ]
        )
    )
    return {
        "synthetic_darts_pick_acc": acc_pick,
        "synthetic_darts_avg_acc": acc_rand,
        "synthetic_darts_gain": acc_pick - acc_rand,
        "synthetic_darts_pick_h": float(hidds[pick]),
        "synthetic_torch_available": 1.0,
    }
