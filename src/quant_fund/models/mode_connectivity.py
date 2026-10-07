"""Linear mode connectivity — loss barrier along the line between two
independently trained minima vs same-basin endpoints; measures whether
the regime task's loss landscape has disconnected basins.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._td_synth import make_data, train_mlp


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mode_connectivity requires torch (pip install -e .[nn])") from exc
    return torch


def bench_mode_connectivity(seed: int = 2371) -> dict[str, float]:
    torch = _torch()
    X, y, _, _ = make_data(seed)
    na, _ = train_mlp(torch, X, y, iters=400, seed=seed)
    nb, _ = train_mlp(torch, X, y, iters=400, seed=seed + 77)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()[:, None]
    pa = [p.detach().clone() for p in na.parameters()]
    pb = [p.detach().clone() for p in nb.parameters()]
    barriers = []
    for al in np.linspace(0, 1, 11):
        with torch.no_grad():
            for p, a, b in zip(nb.parameters(), pa, pb, strict=True):
                p.copy_((1 - al) * a + al * b)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(nb(Xt), yt)
        barriers.append(float(loss))
    endpoints = 0.5 * (barriers[0] + barriers[-1])
    barrier = max(barriers) - endpoints
    return {
        "synthetic_lmc_barrier": float(barrier),
        "synthetic_lmc_endpoint_loss": float(endpoints),
        "synthetic_lmc_barrier_rel": float(barrier / max(endpoints, 1e-6)),
        "synthetic_torch_available": 1.0,
    }
