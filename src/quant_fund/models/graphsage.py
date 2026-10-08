"""GraphSAGE — sample-and-aggregate inductive node embedding.

Hamilton et al. 2017: h_v <- act(W [h_v || AGG({h_u : u in N(v)})]) with a
learned aggregator (mean here); representations are inductive — features +
local neighborhood suffice, no global Laplacian. Bench on the shared SBM
fixture vs graph-blind MLP. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261022


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _mean_agg(adj: FloatArray) -> FloatArray:
    """Row-normalized adjacency for mean aggregation (no self loop in agg)."""
    deg = np.maximum(adj.sum(1), 1e-9)
    return (adj / deg[:, None]).astype(np.float64)


def bench_graphsage(
    seed: int = 5,
    n_train_frac: float = 0.35,
    iters: int = 500,
    d_hid: int = 32,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    adj, x, y = synth_sbm_graph(seed=seed + _SEED)
    agg = _mean_agg(adj)
    tr, te = split_masks(x.shape[0], n_train_frac, seed + _SEED)
    k_cls = int(y.max()) + 1

    at = torch.tensor(agg, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_m = torch.tensor(tr)
    te_m = torch.tensor(te)

    w_self = torch.nn.Linear(x.shape[1], d_hid, bias=False)
    w_neigh = torch.nn.Linear(x.shape[1], d_hid, bias=False)
    w2_self = torch.nn.Linear(d_hid, d_hid, bias=False)
    w2_neigh = torch.nn.Linear(d_hid, d_hid, bias=False)
    w_out = torch.nn.Linear(d_hid, k_cls)
    params = (
        list(w_self.parameters())
        + list(w_neigh.parameters())
        + list(w2_self.parameters())
        + list(w2_neigh.parameters())
        + list(w_out.parameters())
    )
    opt = torch.optim.Adam(params, lr=5e-3)

    def sage_forward(xb: Any) -> Any:
        h = torch.nn.functional.relu(w_self(xb) + w_neigh(at @ xb))
        h = w2_self(h) + w2_neigh(at @ h)
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
        loss = torch.nn.functional.cross_entropy(sage_forward(xt)[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_m = torch.nn.functional.cross_entropy(mlp(xt)[tr_m], yt[tr_m])
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()

    with torch.no_grad():
        acc_sage = float((sage_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_mlp = float((mlp(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
    return {
        "synthetic_sage_acc": acc_sage,
        "synthetic_sage_mlp_acc": acc_mlp,
        "synthetic_sage_acc_gain": acc_sage - acc_mlp,
        "synthetic_torch_available": 1.0,
    }
