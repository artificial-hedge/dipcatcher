"""Fundamental group of the circle: winding numbers classify loops (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def winding_number(path: np.ndarray) -> int:
    """Winding number of a closed path in R^2 minus the origin via angle increments."""
    ang = np.unwrap(np.arctan2(path[:, 1], path[:, 0]))
    return int(np.round((ang[-1] - ang[0]) / (2 * np.pi)))


def loop_s1(n_wind: int, steps: int = 2000) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * n_wind, steps)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def _bench_homotopy_pi1(seed: int = 0) -> float:
    checks = []
    checks.append(winding_number(loop_s1(1)) == 1)
    checks.append(winding_number(loop_s1(3)) == 3)
    checks.append(winding_number(loop_s1(-2)) == -2)
    # concatenated loops add winding: wind(f*g) = wind f + wind g
    checks.append(
        winding_number(loop_s1(2)) + winding_number(loop_s1(3)) == winding_number(loop_s1(5))
    )
    # a loop that goes out and back has winding 0 (homotopic to constant)
    t = np.linspace(0, np.pi, 1000)
    out = np.stack([np.cos(t), np.sin(t)], axis=1)
    back = out[::-1]
    checks.append(winding_number(np.vstack([out, back])) == 0)
    # perturbed circle still winds once (homotopy invariance)
    rng = np.random.default_rng(seed)
    base = loop_s1(1)
    pert = base + 0.05 * rng.normal(size=base.shape)
    checks.append(winding_number(pert) == 1)
    return float(sum(checks) / len(checks))


def bench_homotopy_pi1(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_pi1": _bench_homotopy_pi1(seed)}
