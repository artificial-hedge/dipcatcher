"""Straight-through estimator (Bengio et al. 2013) — forward uses the
hard top-k mask; backward substitutes the identity gradient. Gradient
direction vs finite-diff ground truth on the selection task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import TARGET, K, W, finite_diff_grad, grad_corr, sel_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("st_estimator requires torch (pip install -e .[nn])") from exc
    return torch


def bench_st_estimator(seed: int = 2389, trials: int = 20) -> dict[str, float]:
    torch = _torch()
    corrs = []
    for i in range(trials):
        s0, _ = sel_data(seed + i)
        s = torch.tensor(s0).float().requires_grad_(True)
        m_soft = torch.softmax(s * 20, -1)  # sharp proxy gradient
        idx = torch.topk(s, K).indices
        hard = torch.zeros_like(s)
        hard[idx] = 1.0
        mask = hard.detach() + m_soft - m_soft.detach()  # STE
        v = torch.tensor(W).float() @ (mask * s)
        loss = (v - TARGET) ** 2
        loss.backward()
        g_ste = s.grad.numpy()
        g_fd = finite_diff_grad(s0)
        corrs.append(grad_corr(g_ste, g_fd))
    return {
        "synthetic_ste_grad_corr": float(np.mean(corrs)),
        "synthetic_fd_norm": float(np.linalg.norm(finite_diff_grad(sel_data(seed)[0]))),
        "synthetic_torch_available": 1.0,
    }
