"""ApproxNDCG (Bruch et al. 2019) — differentiable NDCG via soft (SYNTHETIC)
approximate ranks: rank_j(s_i) ≈ 1 + Σ_k sigmoid((s_i - s_k)/τ).
"""

from __future__ import annotations

from typing import Any

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("approx_ndcg_ltr requires torch (pip install -e .[nn])") from exc
    return torch


def _soft_ndcg(s: Any, gains: Any, tau: float, torch: Any) -> Any:
    """Soft NDCG per query. The ideal DCG sorts gains descending — an
    unsorted ideal lets the ratio exceed 1."""
    # soft ranks: r_ij = 1 + sum_k sigmoid((s_j - s_k)/tau) for doc j in query i
    diff = (s[:, :, None] - s[:, None, :]) / tau
    ranks = 1.0 + torch.sigmoid(-diff).sum(-1)  # position = 1 + #docs that beat j
    disc = 1.0 / torch.log2(ranks + 1.0)
    ideal_gains = torch.sort(gains, dim=-1, descending=True).values
    pos = torch.arange(2, gains.shape[1] + 2, dtype=torch.float32)
    ideal = (ideal_gains * (1.0 / torch.log2(pos))).sum(-1).clamp_min(1e-9)
    return (gains * disc).sum(-1) / ideal


def bench_approx_ndcg_ltr(
    seed: int = 659,
    iters: int = 120,
    h: int = 32,
    tau: float = 1.0,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = ltr_data(seed=seed)
    xt = torch.tensor(x_tr).float()
    gains = torch.tensor(2.0**y_tr - 1).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    for _i in range(iters):
        s = net(xt).squeeze(-1)  # (q,doc)
        ndcg = _soft_ndcg(s, gains, tau, torch)
        loss = -ndcg.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sc = net(torch.tensor(x_te).float()).squeeze(-1).numpy()
    nd = ndcg_at(y_te, sc)
    nd_base = ndcg_at(y_te, x_te.mean(-1))
    return {
        "synthetic_andcg_ndcg10": nd,
        "synthetic_andcg_base_ndcg10": nd_base,
        "synthetic_andcg_gain": nd - nd_base,
        "synthetic_torch_available": 1.0,
    }
