"""Frank-Wolfe conditional gradient over polytopes (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def fw_minimize(
    grad_f,
    vertices: np.ndarray,
    x0: np.ndarray,
    iters: int = 300,
) -> np.ndarray:
    """FW over conv(vertices): the LMO solves min <g, v> over the vertex set."""
    x = x0.copy()
    for k in range(iters):
        g = grad_f(x)
        dots = vertices @ g
        s = vertices[int(np.argmin(dots))]
        gamma = 2.0 / (k + 2)
        x = x + gamma * (s - x)
    return x


def _bench_frank_wolfe2(seed: int = 0) -> float:
    checks = []
    # min ||x - t||^2 over the 0.5-scaled L1 ball (octahedron)
    t = np.array([0.8, 0.3, -0.2])
    verts = np.concatenate([np.eye(3), -np.eye(3)]) * 0.5
    grad = lambda x: 2 * (x - t)  # noqa: E731
    x = fw_minimize(grad, verts, np.array([0.5, 0.0, 0.0]), iters=400)
    # t has L1 norm 1.1 > 0.5 so the optimum lies on the boundary
    checks.append(abs(np.abs(x).sum() - 0.5) < 0.05)
    # descent vs start
    checks.append(np.linalg.norm(x - t) < np.linalg.norm(np.array([0.5, 0, 0]) - t) + 0.2)
    # min of a linear function over the simplex picks the argmin vertex
    verts2 = np.eye(4)
    c = np.array([1.0, 0.5, -2.0, 0.0])
    grad2 = lambda x: c  # noqa: E731
    x2 = fw_minimize(grad2, verts2, np.array([0.25, 0.25, 0.25, 0.25]), iters=500)
    checks.append(x2[2] > 0.95)
    # quadratic on the standard 2-simplex: t=(0.3,0.7) is itself a convex
    # combo of the vertices, so FW converges to it exactly
    verts3 = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    grad3 = lambda x: 2 * (x - np.array([0.3, 0.7]))  # noqa: E731
    x3 = fw_minimize(grad3, verts3, np.array([0.0, 0.0]), iters=600)
    checks.append(np.linalg.norm(x3 - np.array([0.3, 0.7])) < 0.15)
    return float(sum(checks) / len(checks))


def bench_frank_wolfe2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frank_wolfe2": _bench_frank_wolfe2(seed)}
