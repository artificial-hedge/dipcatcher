"""Gumbel-top-k differentiable subset selection (Kool et al. 2019) (SYNTHETIC).

Sequential Gumbel-without-replacement: at each pick, add Gumbel noise to
log-probs and take a soft argmax with temperature τ — a k-hot stochastic
relaxation. On the masked-feature fixture the learned selector recovers
the true signal dims vs a logistic-regression |w| baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cert_synth import synth_subset

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("gumbel_topk needs the torch `nn` extra") from exc


def bench_gumbel_topk(
    seed: int = 117,
    n_train: int = 400,
    m: int = 12,
    k: int = 3,
    iters: int = 600,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, _y, mask = synth_subset(n_train, m, k, rng)
    y = (np.prod(np.sign(x[:, :k]) + 1e-9, axis=1) > 0).astype(np.int64)
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    logits = torch.nn.Parameter(torch.zeros(m))
    head = torch.nn.Sequential(torch.nn.Linear(k, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    opt = torch.optim.Adam([logits] + list(head.parameters()), lr=5e-3)
    tau = 0.4
    for _i in range(iters):
        logits_r = logits[None, :].repeat(x_t.shape[0], 1)
        picks = []
        mask_l = torch.zeros(x_t.shape[0], m)
        for _j in range(k):
            g = -torch.log(-torch.log(torch.rand_like(logits_r) + 1e-9) + 1e-9)
            p = torch.softmax((logits_r + g) / tau - 1e9 * mask_l, -1)
            picks.append(p)
            idx = p.argmax(-1)
            mask_l = mask_l.scatter(1, idx[:, None], 1.0)
        sel = torch.stack(picks, 1)
        feats = (x_t[:, None, :] * sel).sum(-1)
        loss = torch.nn.functional.cross_entropy(head(feats), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        topk = logits.argsort(descending=True)[:k].numpy()
    prec = float(np.isin(topk, np.where(mask)[0]).mean())
    w_lr = torch.nn.Linear(m, 2)
    opt3 = torch.optim.Adam(w_lr.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(w_lr(x_t), y_t)
        opt3.zero_grad()
        loss.backward()
        opt3.step()
    with torch.no_grad():
        w_sel = w_lr.weight.norm(dim=0).argsort(descending=True)[:k].numpy()
    prec_lr = float(np.isin(w_sel, np.where(mask)[0]).mean())
    return {
        "synthetic_gtopk_precision": prec,
        "synthetic_gtopk_lr_precision": prec_lr,
        "synthetic_gtopk_gain": prec - prec_lr,
        "synthetic_torch_available": 1.0,
    }
