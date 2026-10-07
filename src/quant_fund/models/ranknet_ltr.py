"""RankNet (Burges et al. 2005) — pairwise logistic on doc pairs (SYNTHETIC).

Scores with an MLP; pairwise BCE weighted by label difference —
NDCG@10 vs unsupervised feature-mean baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ranknet_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_ranknet_ltr(
    seed: int = 641,
    iters: int = 150,
    h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    yt = torch.tensor(y_tr).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    n_q, n_doc, _d = x_tr.shape
    rng = np.random.default_rng(seed)
    for _i in range(iters):
        q = rng.integers(n_q)
        s = net(xt[q]).squeeze(-1)
        i, j = np.triu_indices(n_doc, 1)
        dij = yt[q, i] - yt[q, j]
        keep = np.abs(dij.numpy()) > 0
        if keep.sum() == 0:
            continue
        i, j = i[keep], j[keep]
        lbl = (dij[keep] > 0).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s[i] - s[j], lbl)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc)
    nd_base = ndcg_at(y_te, x_te.mean(-1))
    return {
        "synthetic_ranknet_ndcg10": nd,
        "synthetic_ranknet_base_ndcg10": nd_base,
        "synthetic_ranknet_gain": nd - nd_base,
        "synthetic_torch_available": 1.0,
    }
