"""Graph U-Net — top-k pooling + graph upsampling skip connections.

Gao & Ji 2019: gPool keeps the top-scoring nodes by a learned projection,
gUnpool restores them; U-shaped pooling/unpooling coarsens then refines the
graph while preserving skip connections. Bench: node classification on the
shared SBM fixture vs graph-blind MLP. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import normalize_adj, split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261024
_POOL_RATIO = 0.6


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_graph_unet(
    seed: int = 9,
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
    n = x.shape[0]

    at = torch.tensor(a_norm, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_m = torch.tensor(tr)
    te_m = torch.tensor(te)

    gcn1 = torch.nn.Linear(x.shape[1], d_hid)
    pool_score = torch.nn.Linear(d_hid, 1)
    gcn2 = torch.nn.Linear(d_hid, d_hid)
    gcn3 = torch.nn.Linear(d_hid, d_hid)
    w_out = torch.nn.Linear(d_hid, k_cls)
    params = (
        list(gcn1.parameters())
        + list(pool_score.parameters())
        + list(gcn2.parameters())
        + list(gcn3.parameters())
        + list(w_out.parameters())
    )
    opt = torch.optim.Adam(params, lr=5e-3)

    k_pool = max(int(n * _POOL_RATIO), k_cls * 2)

    def unet_forward(xb: Any) -> Any:
        h1 = torch.nn.functional.relu(gcn1(at @ xb))  # (n, dh)
        # gPool: top-k by learned score, gate h by sigmoid(score)
        s = pool_score(h1).reshape(-1)
        top = torch.topk(s, k_pool).indices
        a_pool = at[top][:, top]
        h_pool = h1[top] * torch.sigmoid(s[top])[:, None]
        h2 = torch.nn.functional.relu(gcn2(a_pool @ h_pool))  # (k, dh)
        # gUnpool: place back at pooled positions, zero elsewhere + skip h1
        h_up = torch.zeros_like(h1)
        h_up[top] = h2
        h3 = torch.nn.functional.relu(gcn3(at @ (h_up + h1)))
        return w_out(h3)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, k_cls),
    )
    opt_m = torch.optim.Adam(mlp.parameters(), lr=5e-3)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(unet_forward(xt)[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_m = torch.nn.functional.cross_entropy(mlp(xt)[tr_m], yt[tr_m])
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()

    with torch.no_grad():
        acc_unet = float((unet_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_mlp = float((mlp(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
    return {
        "synthetic_gunet_acc": acc_unet,
        "synthetic_gunet_mlp_acc": acc_mlp,
        "synthetic_gunet_acc_gain": acc_unet - acc_mlp,
        "synthetic_gunet_pool_size": float(k_pool),
        "synthetic_torch_available": 1.0,
    }
