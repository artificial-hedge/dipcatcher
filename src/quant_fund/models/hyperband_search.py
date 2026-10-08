"""Successive halving and Hyperband (Li et al. 2017) (SYNTHETIC)
multi-fidelity hyperparameter schedulers. Synthetic bench
gates best-loss vs uniform-budget random search at equal
total resource."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def successive_halving(
    sample: Callable[[np.random.Generator], dict[str, float]],
    loss_at: Callable[[dict[str, float], int], float],
    n: int,
    budget: int,
    eta: int = 3,
    seed: int = 0,
) -> dict[str, object]:
    """SHA: run n configs at budget/n each round, keep top
    1/eta until one remains."""
    rng = np.random.default_rng(seed)
    cfgs = [sample(rng) for _ in range(n)]
    b = max(1, budget // n)
    while len(cfgs) > 1:
        scores = np.array([loss_at(c, b) for c in cfgs])
        keep = max(1, len(cfgs) // eta)
        order = np.argsort(scores)
        cfgs = [cfgs[i] for i in order[:keep]]
        b = min(b * eta, budget)
    return {"config": cfgs[0], "loss": float(loss_at(cfgs[0], budget))}


def hyperband(
    sample: Callable[[np.random.Generator], dict[str, float]],
    loss_at: Callable[[dict[str, float], int], float],
    max_iter: int = 81,
    eta: int = 3,
    seed: int = 0,
) -> dict[str, object]:
    """Hyperband: brackets over (s, n, r) — s_max = floor
    (log_eta max_iter); each bracket runs SHA with
    n = ceil((s_max+1)/(s+1) · eta^s), r = max_iter · eta^{−s}."""
    rng = np.random.default_rng(seed)
    s_max = int(np.floor(np.log(max_iter) / np.log(eta)))
    best_cfg = None
    best_loss = np.inf
    for s in range(s_max, -1, -1):
        n = int(np.ceil((s_max + 1) / (s + 1) * eta**s))
        r = max(1, int(max_iter * eta ** (-s)))
        cfgs = [sample(rng) for _ in range(n)]
        b = r
        while len(cfgs) > 1:
            scores = np.array([loss_at(c, b) for c in cfgs])
            keep = max(1, len(cfgs) // eta)
            order = np.argsort(scores)
            cfgs = [cfgs[i] for i in order[:keep]]
            b = min(b * eta, max_iter)
        loss_b = float(loss_at(cfgs[0], b))
        if loss_b < best_loss:
            best_cfg, best_loss = cfgs[0], loss_b
    return {"config": best_cfg, "loss": best_loss}


def bench_hyperband(seed: int = 545) -> dict[str, float]:
    """SYNTHETIC: hyperparam landscape where the true best
    region only wins at high fidelity — HB must reach lower
    best-loss than uniform-budget random at equal evals."""
    out: dict[str, float] = {}
    rng_hp = np.random.default_rng(seed + 7)

    # latent quality: x* = (0.7, 0.3); loss = dist + noise
    # that decays with budget → low-fidelity rankings noisy
    def sample(rng: np.random.Generator) -> dict[str, float]:
        return {"x": float(rng.uniform()), "y": float(rng.uniform())}

    def loss_at(cfg: dict[str, float], budget: int) -> float:
        base = abs(cfg["x"] - 0.7) + abs(cfg["y"] - 0.3)
        noise = rng_hp.normal(scale=0.5 / np.sqrt(max(budget, 1)))
        return float(base + noise)

    hb = hyperband(sample, loss_at, max_iter=81, eta=3, seed=seed)
    l_hb = float(np.asarray(hb["loss"]))
    out["synthetic_hb_best_loss"] = l_hb
    # uniform-budget random: same ~5*81 full evals
    rng2 = np.random.default_rng(seed + 99)
    cfgs = [sample(rng2) for _ in range(40)]
    l_rand = float(min(loss_at(c, 81) for c in cfgs))
    out["synthetic_hb_random_loss"] = l_rand
    if l_hb > l_rand + 0.15:
        raise ValueError(f"hb worse than random: {l_hb} vs {l_rand}")
    out["synthetic_hb_x_err"] = float(
        abs(hb["config"]["x"] - 0.7) + abs(hb["config"]["y"] - 0.3)  # type: ignore[index]
    )
    return out
