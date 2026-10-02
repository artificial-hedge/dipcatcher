"""ListNet (Cao et al. 2007) — listwise top-1 cross-entropy between
softmax(score) and softmax(relevance gain) distributions.
"""

from __future__ import annotations

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("listnet_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_listnet_ltr(
    seed: int = 643,
    iters: int = 150,
    h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    gains = torch.tensor(2.0**y_tr - 1).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    for _i in range(iters):
        s = net(xt).squeeze(-1)
        p = torch.log_softmax(s, -1)
        q = torch.softmax(gains, -1)
        loss = -(q * p).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc)
    nd_base = ndcg_at(y_te, x_te.mean(-1))
    return {
        "synthetic_listnet_ndcg10": nd,
        "synthetic_listnet_base_ndcg10": nd_base,
        "synthetic_listnet_gain": nd - nd_base,
        "torch_available": 1.0,
    }
