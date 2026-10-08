"""Soft decision tree (Irsoy et al. 2012; Frost-Hinton 2017) (SYNTHETIC).

Every internal node is a logistic gate; leaf probabilities mix —
interpretable routing vs hard tree on the synth task.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("soft_tree requires torch (pip install -e .[nn])") from exc
    return torch


def bench_soft_tree(
    seed: int = 607,
    n: int = 400,
    iters: int = 120,
    depth: int = 3,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_t = torch.tensor(x_tr).float()
    y_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    n_int = 2**depth - 1
    n_leaf = 2**depth
    torch.manual_seed(seed)
    gates = [torch.nn.Linear(8, 1) for _ in range(n_int)]
    leaves = torch.nn.Parameter(torch.randn(n_leaf, 2) * 0.1)
    opt = torch.optim.Adam([p for g in gates for p in g.parameters()] + [leaves], lr=0.02)

    def probs(x):
        # node probabilities via DFS: p_leaf = prod gates
        cols = []
        for leaf in range(n_leaf):
            bits = format(leaf, f"0{depth}b")
            p = torch.ones(len(x))
            node = 0
            for b in bits:
                g = torch.sigmoid(gates[node](x).squeeze(-1))
                p = p * (g if b == "1" else 1 - g)
                node = node * 2 + 1 + int(b)
            cols.append(p)
        return torch.stack(cols, 1)

    for _i in range(iters):
        pred = probs(x_t) @ leaves
        loss = torch.nn.functional.cross_entropy(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_s = float(((probs(x_te_t) @ leaves).argmax(1) == y_te_t).float().mean())
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
        "synthetic_st_acc": acc_s,
        "synthetic_st_mlp_acc": acc_m,
        "synthetic_st_gain": acc_s - acc_m,
        "synthetic_torch_available": 1.0,
    }
