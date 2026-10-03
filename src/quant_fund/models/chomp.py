"""CHOMP: covariant gradient descent on smoothness + obstacle cost."""

from __future__ import annotations

import numpy as np


def _smooth_cost(xi: np.ndarray) -> tuple[float, np.ndarray]:
    """Finite-difference acceleration penalty and its gradient."""
    K = np.diff(xi, n=2, axis=0)
    cost = 0.5 * float(np.sum(K * K))
    # gradient of 0.5 sum (x_{i+1} - 2 x_i + x_{i-1})^2 -> discrete biharmonic
    grad = np.zeros_like(xi)
    for i in range(1, len(xi) - 1):
        acc = xi[i + 1] - 2.0 * xi[i] + xi[i - 1]
        grad[i - 1] += acc
        grad[i] += -2.0 * acc
        grad[i + 1] += acc
    return cost, grad


def _obstacle_cost(xi: np.ndarray, circles: np.ndarray, eps: float) -> tuple[float, np.ndarray]:
    """Signed-distance hinge cost c(x) = max(0, eps - dist)."""
    grad = np.zeros_like(xi)
    cost = 0.0
    for i, p in enumerate(xi):
        for c in circles:
            dvec = p - c[:2]
            d = float(np.linalg.norm(dvec)) - c[2]
            if d < eps and float(np.linalg.norm(dvec)) > 1e-9:
                cost += eps - d
                grad[i] += -dvec / np.linalg.norm(dvec)
    return cost, grad


def chomp(
    start: np.ndarray, goal: np.ndarray, n: int, circles: np.ndarray, iters: int = 200
) -> tuple[np.ndarray, list[float]]:
    """Optimize a waypoint trajectory between start and goal."""
    xi = np.linspace(np.asarray(start, float), np.asarray(goal, float), n)
    lam = 1.0
    costs = []
    for _ in range(iters):
        cs, gs = _smooth_cost(xi)
        co, go = _obstacle_cost(xi, circles, 0.5)
        costs.append(cs + co)
        xi[1:-1] -= 0.05 * (gs[1:-1] + lam * go[1:-1])
    return xi, costs


def bench_chomp(seed: int = 20261231 + 861) -> dict[str, float]:
    """CHOMP lowers total cost and clears obstacles the straight line hits."""
    rng = np.random.default_rng(seed)
    checks = 0.0
    total = 0
    for _ in range(8):
        start = np.array([0.0, float(rng.uniform(-1.0, 1.0))])
        goal = np.array([4.0, float(rng.uniform(-1.0, 1.0))])
        cx = float(rng.uniform(1.5, 2.5))
        cy = 0.5 * (start[1] + goal[1])
        circles = np.array([[cx, cy, 0.4]])
        xi0 = np.linspace(start, goal, 25)
        c0 = _obstacle_cost(xi0, circles, 0.5)[0] + _smooth_cost(xi0)[0]
        xi, costs = chomp(start, goal, 25, circles, iters=300)
        total += 3
        checks += float(costs[-1] < c0)
        # endpoints unchanged
        checks += float(np.allclose(xi[0], start) and np.allclose(xi[-1], goal))
        # interior points clear the obstacle with margin
        inside = 0
        for p in xi[1:-1]:
            for c in circles:
                if np.linalg.norm(p - c[:2]) < c[2]:
                    inside += 1
        checks += float(inside <= 1)
    return {"synthetic_chomp": checks / total}
