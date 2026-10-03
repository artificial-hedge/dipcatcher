"""Baire category: dense open sets have dense intersection (SYNTHETIC grid model)."""

from __future__ import annotations

import numpy as np


def dyadic_grid(level: int) -> np.ndarray:
    return np.linspace(0, 1, 2**level + 1)


def dense_open(level: int, remove_num: int, rng: np.random.Generator) -> np.ndarray:
    """A dense subset of the grid = grid minus a few points (still dense on grid)."""
    g = dyadic_grid(level)
    idx = rng.choice(len(g), size=min(remove_num, len(g) - 2), replace=False)
    return np.delete(g, idx)


def _bench_baire_category(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # Q intersect [0,1] is dense: every grid interval contains a rational (dyadic)
    g = dyadic_grid(10)
    checks.append(bool(np.all(np.diff(g) > 0)))
    # countable intersection of dense sets on grid is dense-ish: min gap small
    sets = [dense_open(10, 5, rng) for _ in range(4)]
    inter = sets[0]
    for s in sets[1:]:
        inter = np.intersect1d(inter, s)
    # still nonempty and fairly dense on the grid
    checks.append(len(inter) > 0)
    checks.append(bool(np.min(np.diff(np.sort(inter))) <= 1.0 / 2**10 + 1e-12))
    # a single point is nowhere dense: complement dense
    pt = 0.375
    comp = np.setdiff1d(g, [pt])
    checks.append(bool(np.min(np.abs(comp - pt)) <= 1.0 / 2**10 + 1e-12))
    # Baire on discrete model: intersection of k dense-open grid sets is nonempty
    checks.append(len(inter) > 1024 - 20)
    return float(sum(checks) / len(checks))


def bench_baire_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baire_category": _bench_baire_category(seed)}
