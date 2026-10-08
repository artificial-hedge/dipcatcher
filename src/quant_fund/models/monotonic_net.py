"""Monotonic-by-construction network (min-max lattice-style) (SYNTHETIC).

Outputs are a min over max-pools of affine terms with positive weights
(Daniels & Velikova style) — monotone in x by construction; a plain MLP
trained on the same noisy data wiggles and violates monotonicity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._geo_synth import synth_monotonic

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("monotonic_net needs the torch `nn` extra") from exc


def bench_monotonic_net(
    seed: int = 83,
    n_train: int = 300,
    n_probe: int = 200,
    iters: int = 800,
    n_terms: int = 12,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_monotonic(n_train, rng)
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y).float()
    w = torch.nn.Parameter(torch.rand(n_terms, 1) * 0.1)
    c = torch.linspace(-2.5, 2.5, n_terms)[:, None]
    b0 = torch.nn.Parameter(torch.zeros(1, 1))
    mlp = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    params = [w, b0] + list(mlp.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def mono_fwd(xb):
        ww = torch.nn.functional.softplus(w)
        ww = ww / ww.sum()
        z = b0 + (ww.T * torch.sigmoid((xb - c.T) / 0.2)).sum(-1, keepdim=True) * 1.8
        return z

    for _i in range(iters):
        loss = (mono_fwd(x_t) - y_t).pow(2).mean() + (mlp(x_t) - y_t).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    xp = torch.linspace(-4, 4, n_probe)[:, None]
    with torch.no_grad():
        ym = mono_fwd(xp)
        yr = mlp(xp)
    viol_m = float((ym[1:] - ym[:-1] < -1e-6).float().mean())
    viol_r = float((yr[1:] - yr[:-1] < -1e-6).float().mean())
    with torch.no_grad():
        mse_m = float((mono_fwd(x_t) - y_t).pow(2).mean())
        mse_r = float((mlp(x_t) - y_t).pow(2).mean())
    return {
        "synthetic_mono_viol": viol_m,
        "synthetic_mono_mlp_viol": viol_r,
        "synthetic_mono_mse": mse_m,
        "synthetic_mono_mlp_mse": mse_r,
        "synthetic_torch_available": 1.0,
    }
