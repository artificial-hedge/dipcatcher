"""NeuralSort (Grover et al. 2019) — deterministic differentiable
permutation via P[i,j] = softmax(-(|i+1 - 2j - n| s_j - A s_j)/τ);
soft-sorted gains drive a listwise NDCG surrogate.
"""

from __future__ import annotations

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("neural_sort_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def bench_neural_sort_ltr(
    seed: int = 661,
    iters: int = 120,
    h: int = 32,
    tau: float = 0.5,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    gains = torch.tensor(2.0**y_tr - 1).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    n = x_tr.shape[1]
    ar_i = torch.arange(1, n + 1).float()[:, None]
    ar_j = torch.arange(1, n + 1).float()[None, :]
    mask = ar_i - 2 * ar_j - n
    disc = 1.0 / torch.log2(torch.arange(2, n + 2).float())
    for _i in range(iters):
        s = net(xt).squeeze(-1)  # (q,n)
        # P: (q, pos, doc)
        P = torch.softmax(-(mask[None] * s[:, None, :] - s[:, None, :].abs()) / tau, -1)
        sorted_gains = torch.bmm(P, gains[:, :, None]).squeeze(-1)
        ndcg = (sorted_gains * disc[None]).sum(-1).mean()
        loss = -ndcg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc)
    nd_base = ndcg_at(y_te, x_te.mean(-1))
    return {
        "synthetic_nsort_ndcg10": nd,
        "synthetic_nsort_base_ndcg10": nd_base,
        "synthetic_nsort_gain": nd - nd_base,
        "synthetic_torch_available": 1.0,
    }
