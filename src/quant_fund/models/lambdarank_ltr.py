"""LambdaRank-lite (Burges 2006) — pairwise BCE reweighted by |ΔNDCG|
of swapping the pair, the canonical lambda weighting.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("lambdarank_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_lambdarank_ltr(
    seed: int = 653,
    iters: int = 150,
    h: int = 32,
    k: int = 10,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    gains_np = 2.0**y_tr - 1
    disc = 1.0 / np.log2(np.arange(2, y_tr.shape[1] + 2))
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    n_q, n_doc, _d = x_tr.shape
    rng = np.random.default_rng(seed)
    for _i in range(iters):
        q = rng.integers(n_q)
        s = net(xt[q]).squeeze(-1)
        i, j = np.triu_indices(n_doc, 1)
        dij = y_tr[q, i] - y_tr[q, j]
        keep = np.abs(dij) > 0
        if keep.sum() == 0:
            continue
        i, j = i[keep], j[keep]
        # |ΔNDCG| of swapping positions i,j for pairs with different grades
        dndcg = np.abs((gains_np[q, i] - gains_np[q, j]) * (disc[i] - disc[j]))
        w = torch.tensor(dndcg).float()
        lbl = torch.tensor((dij[keep] > 0).astype(float))
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(s[i] - s[j], lbl, reduction="none")
            * w
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc, k)
    nd_base = ndcg_at(y_te, x_te.mean(-1), k)
    return {
        "synthetic_lambda_ndcg10": nd,
        "synthetic_lambda_base_ndcg10": nd_base,
        "synthetic_lambda_gain": nd - nd_base,
        "synthetic_torch_available": 1.0,
    }
