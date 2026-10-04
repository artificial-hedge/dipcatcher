"""Path lifting for the covering R -> S^1 (exponential map) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def lift_path(base_path: np.ndarray) -> np.ndarray:
    """Lift a loop in S^1 to a path in R via continuous argument."""
    ang = np.unwrap(np.arctan2(base_path[:, 1], base_path[:, 0]))
    return np.asarray(ang)


def deck_transform(lift: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(lift + 2 * np.pi * k)


def _bench_covering_lift(seed: int = 0) -> float:
    checks = []
    t = np.linspace(0, 2 * np.pi, 1000)
    loop = np.stack([np.cos(t), np.sin(t)], axis=1)
    lift = lift_path(loop)
    # exp(lift) = base path
    checks.append(np.allclose(np.exp(1j * lift), loop[:, 0] + 1j * loop[:, 1], atol=1e-9))
    # lift of a loop ends at start + 2*pi*deg
    checks.append(abs((lift[-1] - lift[0]) - 2 * np.pi) < 0.01)
    # deck transform preserves projection
    lifted2 = deck_transform(lift, 3)
    checks.append(np.allclose(np.exp(1j * lifted2), np.exp(1j * lift), atol=1e-9))
    # contractible loop lifts to a loop: back-and-forth path has lift ending at 0
    fwd = np.linspace(0, np.pi, 500)
    b = np.concatenate([fwd, fwd[::-1]])
    bl = np.stack([np.cos(b), np.sin(b)], axis=1)
    lft = lift_path(bl)
    checks.append(abs(lft[-1] - lft[0]) < 1e-9)
    # homotopy lifting: endpoint of lift only depends on homotopy class
    lift2 = lift_path(
        loop * 1.3
    )  # scaled still a loop on S^1? scaling moves off circle but angle same
    checks.append(abs((lift2[-1] - lift2[0]) - (lift[-1] - lift[0])) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_covering_lift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_covering_lift": _bench_covering_lift(seed)}
