"""Catapult dynamics (Lewkowycz et al. 2020) — at large-but-subcritical
lr the loss spikes then diverges/converges; measure the transient
spike amplitude vs convergence of a baseline small-lr run.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._td_synth import make_data, train_mlp


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("catapult_phase requires torch (pip install -e .[nn])") from exc
    return torch


def bench_catapult_phase(seed: int = 2377) -> dict[str, float]:
    torch = _torch()
    X, y, _, _ = make_data(seed)
    _, losses_hi = train_mlp(torch, X, y, iters=300, lr=0.12, seed=seed)
    _, losses_lo = train_mlp(torch, X, y, iters=300, lr=0.005, seed=seed)
    lh = np.asarray(losses_hi)
    spike = float(lh.max() - lh[0])
    final_hi, final_lo = float(lh[-50:].mean()), float(np.mean(losses_lo[-50:]))
    return {
        "synthetic_catapult_spike": spike,
        "synthetic_catapult_final_hi": final_hi,
        "synthetic_catapult_final_lo": final_lo,
        "synthetic_catapult_penalty": final_hi - final_lo,
        "torch_available": 1.0,
    }
