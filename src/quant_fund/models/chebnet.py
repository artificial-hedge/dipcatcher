"""ChebNet — Chebyshev-polynomial spectral graph convolution.

Defferrard et al. 2016: filters on the graph Laplacian approximated by order-K
Chebyshev polynomials T_k(L~) X, localized and O(K|E|) — no eigendecomposition.
Bench: node classification on the shared SBM fixture vs graph-blind MLP.
SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261021
_K_CHEB = 3


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _scaled_laplacian(adj: FloatArray) -> FloatArray:
    """L~ = 2L/lambda_max - I, with lambda_max ~ 2 bound."""
    deg = adj.sum(1)
    dinv = np.power(np.maximum(deg, 1e-9), -0.5)
    lap = np.eye(adj.shape[0]) - dinv[:, None] * adj * dinv[None, :]
    return (lap - np.eye(adj.shape[0])).astype(np.float64)


def bench_chebnet(
    seed: int = 3,
    n_train_frac: float = 0.35,
    iters: int = 500,
    d_hid: int = 32,
) -> dict[str, float]:
    torch = _torch()
    adj, x, y = synth_sbm_graph(seed=seed + _SEED)
    l_tilde = _scaled_laplacian(adj)
    tr, te = split_masks(x.shape[0], n_train_frac, seed + _SEED)
    k_cls = int(y.max()) + 1

    lt = torch.tensor(l_tilde, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y)
    tr_m = torch.tensor(tr)
    te_m = torch.tensor(te)

    # ChebNet layer: out = sum_k T_k(L~) X W_k
    w_cheb = [torch.nn.Linear(x.shape[1], d_hid, bias=False) for _ in range(_K_CHEB)]
    w_out = torch.nn.Linear(d_hid, k_cls)
    params = [p for m in w_cheb for p in m.parameters()] + list(w_out.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)

    def cheb_forward(xb: Any) -> Any:
        t0 = xb
        t1 = lt @ xb
        acc = w_cheb[0](t0) + w_cheb[1](t1)
        tk = t1
        tk_prev = t0
        for k in range(2, _K_CHEB):
            tk_next = 2 * (lt @ tk) - tk_prev
            acc = acc + w_cheb[k](tk_next)
            tk_prev, tk = tk, tk_next
        return w_out(torch.nn.functional.relu(acc))

    mlp = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, k_cls),
    )
    opt_m = torch.optim.Adam(mlp.parameters(), lr=5e-3)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(cheb_forward(xt)[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_m = torch.nn.functional.cross_entropy(mlp(xt)[tr_m], yt[tr_m])
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()

    with torch.no_grad():
        acc_cheb = float((cheb_forward(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_mlp = float((mlp(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
    return {
        "synthetic_cheb_acc": acc_cheb,
        "synthetic_cheb_mlp_acc": acc_mlp,
        "synthetic_cheb_acc_gain": acc_cheb - acc_mlp,
        "synthetic_torch_available": 1.0,
    }
