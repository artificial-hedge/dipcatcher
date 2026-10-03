"""Root system A2: 6 roots in hyperplane, crystallographic properties (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

ROOTS = [
    np.array([1.0, -1.0, 0.0]),
    np.array([0.0, 1.0, -1.0]),
    np.array([1.0, 0.0, -1.0]),
    np.array([-1.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 1.0]),
    np.array([-1.0, 0.0, 1.0]),
]


def coroot_string(a: np.ndarray, b: np.ndarray) -> int:
    """<a, b^v> = 2(a.b)/(b.b); crystallographic => integer."""
    return int(round(2.0 * (a @ b) / (b @ b)))


def _bench_root_lattice_a2(seed: int = 0) -> float:
    checks = []
    checks.append(len(ROOTS) == 6)
    # all roots same length
    checks.append(all(np.isclose(r @ r, 2.0) for r in ROOTS))
    # closure under negation
    checks.append(all(any(np.allclose(-r, s) for s in ROOTS) for r in ROOTS))
    # crystallographic: <alpha, beta^v> integers in {0,±1,±2}
    vals = {coroot_string(a, b) for a in ROOTS for b in ROOTS if not np.allclose(a, b)}
    checks.append(vals <= {-2, -1, 0, 1, 2})
    # simple roots a1,a2 with angle 120 deg: cos = -1/2
    checks.append(np.isclose(ROOTS[0] @ ROOTS[1] / 2.0, -0.5))
    # highest root = a1+a2
    checks.append(np.allclose(ROOTS[0] + ROOTS[1], ROOTS[2]))
    return float(sum(checks) / len(checks))


def bench_root_lattice_a2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_root_lattice_a2": _bench_root_lattice_a2(seed)}
