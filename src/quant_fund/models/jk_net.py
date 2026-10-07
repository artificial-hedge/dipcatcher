"""Jumping-Knowledge network — layer-wise jump connections to the output.

Xu et al. 2018: deeper GNNs over-smooth as neighborhoods grow; JK lets each
node mix representations from all depths (max / concat / LSTM-style) so it can
pick its own receptive field. Bench on the shared SBM fixture vs a plain deep
GCN stack (same depth, no jumps) which degrades. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import normalize_adj, split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261026
_DEPTH = 6


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_jk_net(
    seed: int = 13,
    n_train_frac: float = 0.35,
    iters: int = 500,
    d_hid: int = 32,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    adj, x, y = synth_sbm_graph(seed=seed + _SEED)
    a_norm = normalize_adj(adj)
    tr, te = split_masks(x.shape[0], n_train_frac, seed + _SEED)
    k_cls = int(y.max()) + 1

    at = torch.tensor(a_norm, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_m = torch.tensor(tr)
    te_m = torch.tensor(te)

    in_lin = torch.nn.Linear(x.shape[1], d_hid)
    layers = torch.nn.ModuleList([torch.nn.Linear(d_hid, d_hid) for _ in range(_DEPTH)])
    w_out = torch.nn.Linear(_DEPTH * d_hid, k_cls)
    params = list(in_lin.parameters()) + list(layers.parameters()) + list(w_out.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)

    def jk_forward(xb: Any) -> Any:
        h = torch.nn.functional.relu(in_lin(xb))
        hs = []
        for lin in layers:
            h = torch.nn.functional.relu(lin(at @ h))
            hs.append(h)
        return w_out(torch.cat(hs, -1))  # concat-jump: all depths to output

    # plain deep GCN (no jumps) — over-smoothing comparator
    deep_lin = torch.nn.Linear(x.shape[1], d_hid)
    deep_layers = torch.nn.ModuleList([torch.nn.Linear(d_hid, d_hid) for _ in range(_DEPTH)])
    deep_out = torch.nn.Linear(d_hid, k_cls)
    params_d = (
        list(deep_lin.parameters()) + list(deep_layers.parameters()) + list(deep_out.parameters())
    )
    opt_d = torch.optim.Adam(params_d, lr=5e-3)

    def deep_forward(xb: Any) -> Any:
        h = torch.nn.functional.relu(deep_lin(xb))
        for lin in deep_layers:
            h = torch.nn.functional.relu(lin(at @ h))
        return deep_out(h)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(jk_forward(xt)[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_d = torch.nn.functional.cross_entropy(deep_forward(xt)[tr_m], yt[tr_m])
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

    with torch.no_grad():
        acc_jk = float((jk_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_deep = float((deep_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        # over-smoothing signature: row-variance of final-layer embeddings
        h = torch.nn.functional.relu(in_lin(xt))
        for lin in layers:
            h = torch.nn.functional.relu(lin(at @ h))
        row_var = float(h.var(0).mean())
    return {
        "synthetic_jk_acc": acc_jk,
        "synthetic_jk_deep_acc": acc_deep,
        "synthetic_jk_acc_gain": acc_jk - acc_deep,
        "synthetic_jk_emb_rowvar": row_var,
        "synthetic_torch_available": 1.0,
    }
