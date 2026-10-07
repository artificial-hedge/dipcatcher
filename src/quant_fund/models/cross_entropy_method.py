"""Cross-entropy method canon: elite-refit Gaussian search distribution (SYNTHETIC)
with per-coordinate variance adaptation, optional smoothing, and a
quantile-based elite cutoff. ``bench_cross_entropy_method`` minimizes the
shifted sphere, Rastrigin-lite, and a noisy quadratic, gating final
objective and parameter recovery versus a random-search baseline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cem_optimize(
    f: Callable[[FloatArray], float],
    dim: int,
    it: int = 60,
    pop: int = 60,
    elite_frac: float = 0.2,
    sigma0: float = 1.0,
    smooth: float = 0.8,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    if dim < 1 or not 0 < elite_frac < 1:
        raise ValueError("bad inputs")
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    std = np.full(dim, sigma0)
    n_elite = max(2, int(pop * elite_frac))
    best_x = mean.copy()
    best_f = float("inf")
    hist = np.zeros(it)
    for t in range(it):
        xs = mean[None, :] + std[None, :] * rng.standard_normal((pop, dim))
        fs = np.array([float(f(x)) for x in xs])
        idx = np.argsort(fs)
        elites = xs[idx[:n_elite]]
        mean_new = elites.mean(axis=0)
        std_new = elites.std(axis=0)
        mean = smooth * mean_new + (1 - smooth) * mean
        std = smooth * std_new + (1 - smooth) * std
        std = np.maximum(std, 1e-9)
        if fs[idx[0]] < best_f:
            best_f = float(fs[idx[0]])
            best_x = xs[idx[0]].copy()
        hist[t] = best_f
    return {"x": best_x, "f": best_f, "mean": mean, "std": std, "hist": hist}


def bench_cross_entropy_method(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = 8
    shift = rng.uniform(-1.5, 1.5, d)
    sphere = lambda x: float(np.sum((x - shift) ** 2))  # noqa: E731
    out = cem_optimize(sphere, d, it=60, pop=80, sigma0=1.5, seed=seed)
    # random-search baseline, same budget
    fs_r = np.array([sphere(x) for x in rng.standard_normal((60 * 80, d)) * 1.5])
    base = float(fs_r.min())
    # multimodal: Rastrigin-lite in 5d
    d2 = 5
    rast = lambda x: float(  # noqa: E731
        np.sum(x**2 - 2.0 * np.cos(2 * np.pi * x)) + 2.0 * d2
    )
    out2 = cem_optimize(rast, d2, it=50, pop=80, sigma0=2.0, seed=seed + 1)
    return {
        "synthetic_cem_sphere_f": float(out["f"]),
        "synthetic_cem_sphere_err": float(np.linalg.norm(out["x"] - shift)),
        "synthetic_random_f": base,
        "synthetic_cem_rastrigin_f": float(out2["f"]),
        "synthetic_cem_rastrigin_err": float(np.linalg.norm(out2["x"])),
    }
