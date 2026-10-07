"""APPNP — personalized-PageRank propagation on top of a learned encoder.

Klicpera et al. 2019: decouple prediction from propagation — a shallow net
produces logits, then Z = alpha * (I - (1-alpha) A_hat)^{-1} * logits diffuses
them via personalized PageRank (iterated power steps). Fixes over-smoothing
by keeping the teleport-to-self term. Bench on the shared SBM fixture vs the
same encoder WITHOUT propagation. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._graph_synth import normalize_adj, split_masks, synth_sbm_graph

FloatArray = NDArray[np.float64]

_SEED = 20261025
_ALPHA = 0.12  # teleport prob — PPR restart
_PPROP_STEPS = 10


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_apnp_prop(
    seed: int = 11,
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

    enc = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, k_cls),
    )
    opt = torch.optim.Adam(enc.parameters(), lr=5e-3)
    enc_blind = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], d_hid),
        torch.nn.ReLU(),
        torch.nn.Linear(d_hid, k_cls),
    )
    opt_b = torch.optim.Adam(enc_blind.parameters(), lr=5e-3)

    def pprop(z0: Any) -> Any:
        z = z0
        for _ in range(_PPROP_STEPS):
            z = (1 - _ALPHA) * (at @ z) + _ALPHA * z0
        return z

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(pprop(enc(xt))[tr_m], yt[tr_m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_b = torch.nn.functional.cross_entropy(enc_blind(xt)[tr_m], yt[tr_m])
        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

    with torch.no_grad():
        acc_apnp = float((pprop(enc(xt)).argmax(-1)[te_m] == yt[te_m]).float().mean())
        acc_blind = float((enc_blind(xt).argmax(-1)[te_m] == yt[te_m]).float().mean())
        # over-smoothing check: propagated logits retain node-identity spread
        z = pprop(enc(xt))
        spread = float(z.std(0).mean())
    return {
        "synthetic_apnp_acc": acc_apnp,
        "synthetic_apnp_blind_acc": acc_blind,
        "synthetic_apnp_acc_gain": acc_apnp - acc_blind,
        "synthetic_apnp_logit_spread": spread,
        "synthetic_torch_available": 1.0,
    }
