"""MoE router (Shazeer et al. 2017) — top-2 noisy-gated experts with
load-balancing aux loss on the 4-regime task vs a dense MLP of equal
capacity. Experts should specialize to regimes.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import regime_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("moe_router requires torch (pip install -e .[nn])") from exc
    return torch


def _train_moe(seed: int, iters: int = 700, k: int = 4) -> tuple[float, float]:
    torch = _torch()
    X, y = regime_task(seed, k=k)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    torch.manual_seed(seed)
    D = 16
    gate = torch.nn.Linear(8, k)
    experts = torch.nn.ModuleList(
        [
            torch.nn.Sequential(torch.nn.Linear(8, D), torch.nn.ReLU(), torch.nn.Linear(D, 1))
            for _ in range(k)
        ]
    )
    opt = torch.optim.Adam(list(gate.parameters()) + list(experts.parameters()), lr=0.01)
    for _ in range(iters):
        logits = gate(Xt) + 0.3 * torch.randn_like(gate(Xt))
        w = torch.softmax(logits, -1)
        top2 = torch.topk(w, 2, -1)
        mask = torch.zeros_like(w).scatter(1, top2.indices, top2.values)
        mask = mask / mask.sum(-1, keepdim=True).clamp_min(1e-6)
        out = torch.stack([e(Xt).squeeze(-1) for e in experts], -1)
        pred = (mask * out).sum(-1)
        lb = (w.mean(0) * k).pow(2).sum()  # load-balance aux
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt) + 0.01 * lb
        opt.zero_grad()
        loss.backward()
        opt.step()
    X2, y2 = regime_task(seed + 1, k=k)
    Xt2 = torch.tensor(X2).float()
    with torch.no_grad():
        w = torch.softmax(gate(Xt2), -1)
        top2 = torch.topk(w, 2, -1)
        mask = torch.zeros_like(w).scatter(1, top2.indices, top2.values)
        mask = mask / mask.sum(-1, keepdim=True).clamp_min(1e-6)
        out = torch.stack([e(Xt2).squeeze(-1) for e in experts], -1)
        pred = (mask * out).sum(-1)
    acc = float(((pred > 0).float().numpy() == y2).mean())
    util = float(w.mean(0).min() / w.mean(0).mean())  # expert balance
    return acc, util


def _train_dense(seed: int, iters: int = 700) -> float:
    torch = _torch()
    X, y = regime_task(seed)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(-1), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    X2, y2 = regime_task(seed + 1)
    Xt2 = torch.tensor(X2).float()
    return float(((net(Xt2).squeeze(-1) > 0).float().numpy() == y2).mean())


def bench_moe_router(seed: int = 1727, iters: int = 700) -> dict[str, float]:
    acc_m, util = _train_moe(seed, iters)
    acc_d = _train_dense(seed + 1, iters)
    return {
        "synthetic_moe_acc": acc_m,
        "synthetic_dense_acc": acc_d,
        "synthetic_moe_gain": acc_m - acc_d,
        "synthetic_moe_expert_balance": util,
        "synthetic_torch_available": 1.0,
    }
