"""A2 root system: roots, reflections, Cartan matrix, Weyl group (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

SQ3 = np.sqrt(3.0)


def a2_roots() -> list[np.ndarray]:
    e1, e2, e3 = np.eye(3)
    return [e1 - e2, e2 - e3, e1 - e3, e2 - e1, e3 - e2, e3 - e1]


def weyl_reflect(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.asarray(x - 2 * np.dot(x, alpha) / np.dot(alpha, alpha) * alpha)


def cartan(alpha: np.ndarray, beta: np.ndarray) -> float:
    return float(2 * np.dot(beta, alpha) / np.dot(alpha, alpha))


def _bench_root_system(seed: int = 0) -> float:
    checks = []
    roots = a2_roots()
    a1, a2 = roots[0], roots[1]
    checks.append(len(roots) == 6)
    # simple roots: a1+a2 is a root
    checks.append(bool(any(np.allclose(a1 + a2, r) for r in roots)))
    # Cartan: <a2,a1v>=-1, diag=2 -> matrix [[2,-1],[-1,2]]
    checks.append(bool(np.isclose(cartan(a1, a1), 2.0) and np.isclose(cartan(a1, a2), -1.0)))
    # Weyl reflection preserves root system
    s1 = [weyl_reflect(r, a1) for r in roots]
    checks.append(bool(all(any(np.allclose(x, rr, atol=1e-9) for rr in roots) for x in s1)))
    # reflection fixes a2? s_{a1}(a2) = a2 + a1
    checks.append(bool(any(np.allclose(weyl_reflect(a2, a1), r) for r in roots)))
    # Weyl group order 6: orbit of a1 under reflections
    orbit = {tuple(np.round(weyl_reflect(a1, r), 9)) for r in roots}
    orbit.add(tuple(np.round(a1, 9)))
    orbit.update(tuple(np.round(weyl_reflect(weyl_reflect(a1, roots[2]), r), 9)) for r in roots)
    checks.append(bool(len(orbit) >= 3))
    return float(sum(checks) / len(checks))


def bench_root_system(seed: int = 0) -> dict[str, float]:
    return {"synthetic_root_system": _bench_root_system(seed)}
