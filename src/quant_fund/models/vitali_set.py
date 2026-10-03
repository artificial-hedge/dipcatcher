"""Vitali set: one representative per Q-coset; rational translates tile (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def vitali_set(seed: int = 0, n_grid: int = 4096) -> np.ndarray:
    """Grid Vitali set on [0,1): pick one rep per equivalence class mod rationals
    approximated by mod-(1/n_grid) buckets... we use the standard construction:
    reps of cosets of the subgroup of dyadic-rational shifts on the grid.
    Simplify: classes = sets {x + q mod 1 : q in Q_grid} for a rational grid.
    """
    rng = np.random.default_rng(seed)
    # use q-grid of denominator d; classes are orbits under adding q mod 1
    d = 64
    grid = np.arange(n_grid) / n_grid
    qs = np.arange(d) / d
    seen: set[int] = set()
    reps: list[int] = []
    for i, x in enumerate(grid):
        if i in seen:
            continue
        # orbit of x under +q mod 1 (all land on grid indices when d | n_grid)
        orbit = {int(round((x + q) % 1.0 * n_grid)) % n_grid for q in qs}
        seen |= orbit
        if rng.random() < 1.0:  # pick first rep deterministically = i
            reps.append(i)
    return np.sort(np.array(reps) / n_grid)


def _bench_vitali_set(seed: int = 0) -> float:
    v = vitali_set(seed)
    d = 64
    n_grid = 4096
    checks = []
    # translates V + q mod 1 for q in {k/d} are disjoint
    qs = np.arange(d) / d
    union: set[int] = set()
    disjoint = True
    for q in qs:
        t = np.round((v + q) % 1.0 * n_grid).astype(int) % n_grid
        ts = set(t.tolist())
        if union & ts:
            disjoint = False
        union |= ts
    checks.append(disjoint)
    # translates cover the whole grid (choice property)
    checks.append(len(union) == n_grid)
    # exactly n_grid/d representatives
    checks.append(len(v) == n_grid // d)
    # outer measure can't be 0 (union covers [0,1] up to grid): reps nonempty
    checks.append(len(v) > 0)
    return float(sum(checks) / len(checks))


def bench_vitali_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vitali_set": _bench_vitali_set(seed)}
