"""muP-lite (Yang & Hu 2021) — maximal-update-parameterized scaling:
readout weights init ~ 1/width and lr scaled ~ 1/width, vs standard
init at width 16 and 64; measures whether performance transfers across
widths.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._lm_synth import regime_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mup_init requires torch (pip install -e .[nn])") from exc
    return torch


def _train(seed: int, D: int, mup: bool, iters: int = 500) -> float:
    torch = _torch()
    X, y = regime_task(seed)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    torch.manual_seed(seed)
    W1 = torch.nn.Linear(8, D)
    W2 = torch.nn.Linear(D, 1, bias=False)
    if mup:
        with torch.no_grad():
            W2.weight.mul_(1.0 / D)
        lr = 0.05
    else:
        lr = 0.05 / np.sqrt(D / 16) if D > 16 else 0.05
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=lr)
    for _ in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            W2(torch.relu(W1(Xt))).squeeze(-1), yt
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    X2, y2 = regime_task(seed + 1)
    Xt2 = torch.tensor(X2).float()
    with torch.no_grad():
        out = W2(torch.relu(W1(Xt2))).squeeze(-1)
    return float(((out > 0).float().numpy() == y2).mean())


def bench_mup_init(seed: int = 1733, iters: int = 500) -> dict[str, float]:
    a16 = _train(seed, 16, True, iters)
    a64 = _train(seed, 64, True, iters)
    b16 = _train(seed + 1, 16, False, iters)
    b64 = _train(seed + 1, 64, False, iters)
    return {
        "synthetic_mup_w16": a16,
        "synthetic_mup_w64": a64,
        "synthetic_std_w16": b16,
        "synthetic_std_w64": b64,
        "synthetic_mup_width_gap": abs(a64 - a16),
        "synthetic_std_width_gap": abs(b64 - b16),
        "torch_available": 1.0,
    }
