"""Gumbel-softmax relaxation (Jang et al. 2017; Maddison et al. 2017) —
continuous top-k via temperature-annealed Gumbel-softmax mask; gradient
correlation vs finite-diff on the selection task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import TARGET, K, W, finite_diff_grad, grad_corr, sel_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gumbel_relax requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gumbel_relax(seed: int = 2393, trials: int = 20, tau: float = 0.4) -> dict[str, float]:
    torch = _torch()
    corrs = []
    for i in range(trials):
        s0, _ = sel_data(seed + i)
        s = torch.tensor(s0).float().requires_grad_(True)
        torch.manual_seed(seed + i)
        g = -torch.log(-torch.log(torch.rand(len(s)) + 1e-9) + 1e-9)
        logits = (s + g) / tau
        # top-k relax: normalized weights concentrated on top-k
        m = torch.softmax(logits, -1) * len(s) / K
        m = m.clamp(max=1.0)
        v = torch.tensor(W).float() @ (m * s)
        loss = (v - TARGET) ** 2
        loss.backward()
        g_gs = s.grad.numpy()
        g_fd = finite_diff_grad(s0)
        corrs.append(grad_corr(g_gs, g_fd))
    return {
        "synthetic_gumbel_grad_corr": float(np.mean(corrs)),
        "synthetic_gumbel_tau": float(tau),
        "torch_available": 1.0,
    }
