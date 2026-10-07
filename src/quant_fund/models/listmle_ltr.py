"""ListMLE (Xia et al. 2008) — Plackett-Luce likelihood of the ideal (SYNTHETIC)
ordering: sum over positions of log-softmax on remaining suffix.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("listmle_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_listmle_ltr(
    seed: int = 647,
    iters: int = 150,
    h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    order = torch.tensor(np.argsort(-y_tr, axis=1))
    n_doc = y_tr.shape[1]
    for _i in range(iters):
        s = net(xt).squeeze(-1)
        loss = torch.zeros(1)
        for q in range(len(xt)):
            ideal = order[q]
            loss_q = torch.zeros(1)
            for pos in range(n_doc - 1):
                idx = ideal[pos:]
                scores = s[q][idx]
                loss_q = loss_q - torch.log_softmax(scores, 0)[0]
            loss = loss + loss_q
        loss = loss / len(xt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc)
    nd_base = ndcg_at(y_te, x_te.mean(-1))
    return {
        "synthetic_listmle_ndcg10": nd,
        "synthetic_listmle_base_ndcg10": nd_base,
        "synthetic_listmle_gain": nd - nd_base,
        "synthetic_torch_available": 1.0,
    }
