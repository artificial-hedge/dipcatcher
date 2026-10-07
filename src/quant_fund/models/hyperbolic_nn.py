"""Poincaré-ball embedding layer (Nickel & Kiela 2017) (SYNTHETIC).

Distances via the Poincaré metric; the same leaves embed a random binary
tree at far lower distortion than a Euclidean embedding of the same
dimension — the canonical hyperbolic-geometry advantage on hierarchy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_hierarchy

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("hyperbolic_nn needs the torch `nn` extra") from exc


def _poincare_dist(torch, u, v):
    num = (u - v).pow(2).sum(-1) * 2
    den = (1 - u.pow(2).sum(-1)) * (1 - v.pow(2).sum(-1))
    x = 1 + num / den.clamp_min(1e-4)
    return torch.acosh(x.clamp_min(1 + 1e-6))


def bench_hyperbolic_nn(
    seed: int = 67,
    depth: int = 4,
    iters: int = 800,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    _vecs, _idx, dist = synth_hierarchy(depth, rng)
    n = dist.shape[0]
    emb = torch.nn.Parameter(0.05 * torch.randn(n, 2))
    emb_e = torch.nn.Parameter(0.05 * torch.randn(n, 2))
    opt = torch.optim.Adam([emb], lr=5e-3)
    opt_e = torch.optim.Adam([emb_e], lr=5e-3)
    ii, jj = np.triu_indices(n, 1)
    dt = torch.tensor(dist[ii, jj]).float()
    for _i in range(iters):
        u = emb[ii]
        v = emb[jj]
        d_h = _poincare_dist(torch, u, v)
        loss = ((d_h - dt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            emb.data = emb.data / emb.data.norm(dim=-1, keepdim=True).clamp_min(1 - 1e-3)
    with torch.no_grad():
        d_h = _poincare_dist(torch, emb[ii], emb[jj])
        distortion_h = float(((d_h - dt).abs() / (dt + 1e-9)).mean())
    for _i in range(iters):
        u = emb_e[ii]
        v = emb_e[jj]
        d_e = (u - v).norm(dim=-1)
        loss = ((d_e - dt) ** 2).mean()
        opt_e.zero_grad()
        loss.backward()
        opt_e.step()
    with torch.no_grad():
        d_e = (emb_e[ii] - emb_e[jj]).norm(dim=-1)
        distortion_e = float(((d_e - dt).abs() / (dt + 1e-9)).mean())
    return {
        "synthetic_hyp_distortion": distortion_h,
        "synthetic_hyp_euclid_distortion": distortion_e,
        "synthetic_hyp_gain": float(distortion_e - distortion_h),
        "synthetic_torch_available": 1.0,
    }
