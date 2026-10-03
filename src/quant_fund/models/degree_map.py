"""Degree of a circle map + no-retraction consequence (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def degree_of_map(theta_samples: np.ndarray, images: np.ndarray) -> int:
    """Degree = winding of image curve around 0."""
    ang = np.unwrap(np.arctan2(images[:, 1], images[:, 0]))
    return int(np.round((ang[-1] - ang[0]) / (2 * np.pi)))


def circle_map_image(n: int, power: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(power * t), np.sin(power * t)], axis=1)


def _bench_degree_map(seed: int = 0) -> float:
    checks = []
    n = 2000
    th = np.linspace(0, 2 * np.pi, n)
    checks.append(degree_of_map(th, circle_map_image(n, 1)) == 1)
    checks.append(degree_of_map(th, circle_map_image(n, 3)) == 3)
    checks.append(degree_of_map(th, circle_map_image(n, -2)) == -2)
    # constant map degree 0
    checks.append(degree_of_map(th, np.tile(np.array([1.0, 0.0]), (n, 1))) == 0)
    # deg(f . g) = deg f * deg g
    checks.append(
        degree_of_map(th, circle_map_image(n, 2)) * degree_of_map(th, circle_map_image(n, 3))
        == degree_of_map(th, circle_map_image(n, 6))
    )
    # homotopy invariance: deg preserved under small perturbation
    rng = np.random.default_rng(seed)
    img = circle_map_image(n, 2) * (1 + 0.05 * rng.normal(size=(n, 1)))
    checks.append(degree_of_map(th, img) == 2)
    return float(sum(checks) / len(checks))


def bench_degree_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_degree_map": _bench_degree_map(seed)}
