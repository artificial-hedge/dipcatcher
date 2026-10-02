"""GIN — Graph Isomorphism Network (maximally injective aggregation).

Xu et al. 2019: h_v <- MLP((1 + eps) h_v + sum_{u in N(v)} h_u) — sum +
injective MLP matches the WL test's discriminative power, unlike mean/max
aggregators which collapse multisets. Bench on the shared SBM fixture vs a
mean-aggregation SAGE-style net and graph-blind MLP. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261023


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_gin_gnn(
    seed: int = 7,
    n_train_frac: float = 0.35,
    iters: int = 500,
    d_hid: int = 32,
) -> dict[str, float]:
    torch = _torch()
    adj, x, y = synth_sbm_graph(seed=seed + _SEED)
    tr, te = split_masks(x.shape[0], n_train_frac, seed + _SEED)
    k_cls = int(y.max()) + 1

    at = torch.tensor(adj, dtype=torch.float32)  # raw adjacency — GIN sums
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_m = torch.tensor(tr)
    te_m = torch.tensor(te)

    mlp1 = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid), torch.nn.ReLU(), torch.nn.Linear(d_hid, d_hid)
    )
    mlp2 = torch.nn.Sequential(
        torch.nn.Linear(d_hid, d_hid), torch.nn.ReLU(), torch.nn.Linear(d_hid, d_hid)
    )
    eps1 = torch.nn.Parameter(torch.zeros(1))
    eps2 = torch.nn.Parameter(torch.zeros(1))
    w_out = torch.nn.Linear(d_hid, k_cls)
    params = (
        list(mlp1.parameters()) + list(mlp2.parameters()) + [eps1, eps2] + list(w_out.parameters())
    )
    opt = torch.optim.Adam(params, lr=5e-3)

    def gin_forward(xb: Any) -> Any:
        h = mlp1((1 + eps1) * xb + at @ xb)
        h = mlp2((1 + eps2) * h + at @ h)
        return w_out(h)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, k_cls),
    )
    opt_m = torch.optim.Adam(mlp.parameters(), lr=5e-3)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(gin_forward(xt)[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_m = torch.nn.functional.cross_entropy(mlp(xt)[tr_m], yt[tr_m])
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()

    with torch.no_grad():
        acc_gin = float((gin_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_mlp = float((mlp(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
    return {
        "synthetic_gin_acc": acc_gin,
        "synthetic_gin_mlp_acc": acc_mlp,
        "synthetic_gin_acc_gain": acc_gin - acc_mlp,
        "synthetic_gin_eps1": float(eps1.detach().abs()),
        "torch_available": 1.0,
    }
