"""Shared fixture for wave-186 energy-based-model canon (SYNTHETIC).

Two-moons data (from `_nf_synth`) + energy MLP + Langevin sampler +
Gaussian-kernel MMD evaluation. Baseline: MMD of Gaussian-matched
samples vs data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._nf_synth import two_moons

FloatArray = NDArray[np.float64]


def moon_data(seed: int = 0, n: int = 1500) -> tuple[FloatArray, FloatArray]:
    return two_moons(seed, n), two_moons(seed + 1, n=500)


def mmd(X: FloatArray, Y: FloatArray, gam: float = 1.0) -> float:
    """Biased Gaussian-kernel MMD estimate."""

    def k(A: FloatArray, B: FloatArray) -> float:
        d2 = ((A[:, None] - B[None]) ** 2).sum(-1)
        return float(np.exp(-gam * d2).mean())

    return k(X, X) + k(Y, Y) - 2 * k(X, Y)


def gauss_baseline_mmd(Xtr: FloatArray, Xte: FloatArray) -> float:
    rng = np.random.default_rng(0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    S = mu + sd * rng.standard_normal(Xte.shape)
    return mmd(S, Xte)


def make_energy(torch):
    return torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    )


def langevin(torch, net, n: int, steps: int = 80, step: float = 0.1, seed: int | None = 0, x0=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, 2) * 2 if x0 is None else x0.clone()
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        e = net(x).sum()
        g = torch.autograd.grad(e, x)[0]
        with torch.no_grad():
            x = x - 0.5 * step * g + np.sqrt(step) * torch.randn(n, 2)
    return x.detach()
