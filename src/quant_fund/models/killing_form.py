"""Killing form on sl(2): B(x,y) = tr(ad_x ad_y) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

E = np.array([[0.0, 1.0], [0.0, 0.0]])
F = np.array([[0.0, 0.0], [1.0, 0.0]])
H = np.array([[1.0, 0.0], [0.0, -1.0]])
BASIS = [E, F, H]


def brac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(x @ y - y @ x)


def ad_matrix(x: np.ndarray) -> np.ndarray:
    """3x3 matrix of ad_x in basis E,F,H."""
    cols = []
    for b in BASIS:
        c = brac(x, b)
        # coords of c in basis [E,F,H]
        coords = np.array([c[0, 1], c[1, 0], c[0, 0]])
        cols.append(coords)
    return np.asarray(np.stack(cols, axis=1))


def killing(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trace(ad_matrix(x) @ ad_matrix(y)))


def _bench_killing_form(seed: int = 0) -> float:
    checks = []
    # Killing form of sl2: K(E,F)=4, K(H,H)=8, K(E,H)=0
    checks.append(np.isclose(killing(E, F), 4.0))
    checks.append(np.isclose(killing(H, H), 8.0))
    checks.append(np.isclose(killing(E, H), 0.0))
    checks.append(np.isclose(killing(E, E), 0.0))
    # invariance: K([x,y],z) = K(x,[y,z])
    x = 0.3 * E + 0.5 * F + 0.7 * H
    y = 1.1 * E - 0.4 * F + 0.2 * H
    z = 0.9 * E + 0.3 * F - 0.6 * H
    checks.append(np.isclose(killing(brac(x, y), z), killing(x, brac(y, z))))
    # nondegenerate: gram matrix det != 0
    g = np.array([[killing(BASIS[i], BASIS[j]) for j in range(3)] for i in range(3)])
    checks.append(abs(np.linalg.det(g)) > 1e-9)
    return float(sum(checks) / len(checks))


def bench_killing_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_killing_form": _bench_killing_form(seed)}
