"""NODE-lite — neural oblivious decision ensemble (Popov et al. 2020).

Soft oblivious trees: all leaves share the same decision function
per depth; differentiable routing → tabular accuracy vs MLP.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("node_net requires torch (pip install -e .[nn])") from exc
    return torch


def bench_node_net(
    seed: int = 599,
    n: int = 400,
    iters: int = 120,
    depth: int = 4,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_t = torch.tensor(x_tr).float()
    y_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    torch.manual_seed(seed)
    n_leaf = 2**depth
    # per-depth shared decision: Linear(8→2) then log_softmax routing
    layers = [torch.nn.Linear(8, 2) for _ in range(depth)]
    leaves = torch.nn.Parameter(torch.randn(n_leaf, 2) * 0.1)
    opt = torch.optim.Adam([p for lay in layers for p in lay.parameters()] + [leaves], lr=0.02)
    for _i in range(iters):
        # route: product of per-depth Bernoulli choices → leaf dist
        dist = torch.ones(len(x_t), 1)
        for lay in layers:
            gate = torch.softmax(lay(x_t), -1)  # (n,2) L/R probs
            lft = dist * gate[:, 0:1]
            rgt = dist * gate[:, 1:2]
            dist = torch.cat([lft, rgt], 1)  # splits every leaf
        pred = dist @ leaves  # (n,2) logits
        loss = torch.nn.functional.cross_entropy(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        dist = torch.ones(len(x_te_t), 1)
        for lay in layers:
            gate = torch.softmax(lay(x_te_t), -1)
            lft = dist * gate[:, 0:1]
            rgt = dist * gate[:, 1:2]
            dist = torch.cat([lft, rgt], 1)
        pred = dist @ leaves
        acc_n = float((pred.argmax(1) == y_te_t).float().mean())
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(mlp(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_m = acc_of(torch, mlp, x_te_t, y_te_t)
    return {
        "synthetic_node_acc": acc_n,
        "synthetic_node_mlp_acc": acc_m,
        "synthetic_node_gain": acc_n - acc_m,
        "torch_available": 1.0,
    }
