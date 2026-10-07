"""Smooth-argmax / soft-k-argmax — softmax-weighted aggregation as a (SYNTHETIC)
differentiable argmax proxy; temperature sweep vs hard-argmax on the
selection task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import TARGET, W, finite_diff_grad, grad_corr, sel_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("smooth_argmax requires torch (pip install -e .[nn])") from exc
    return torch


def bench_smooth_argmax(seed: int = 2417, trials: int = 20) -> dict[str, float]:
    torch = _torch()
    corrs: list[float] = []
    corrs_soft: list[float] = []
    for i in range(trials):
        s0, _ = sel_data(seed + i)
        s = torch.tensor(s0).float().requires_grad_(True)
        for tau, out in ((0.1, corrs), (1.0, corrs_soft)):
            m = torch.softmax(s / tau, -1)
            v = torch.tensor(W).float() @ (m * s * len(s))
            loss = (v - TARGET) ** 2
            g = torch.autograd.grad(loss, s)[0].numpy()
            g_fd = finite_diff_grad(s0)
            out.append(grad_corr(g, g_fd))
            s = torch.tensor(s0).float().requires_grad_(True)
    return {
        "synthetic_smax_sharp_corr": float(np.mean(corrs)),
        "synthetic_smax_soft_corr": float(np.mean(corrs_soft)),
        "synthetic_smax_tau_gain": float(np.mean(corrs) - np.mean(corrs_soft)),
        "synthetic_torch_available": 1.0,
    }
