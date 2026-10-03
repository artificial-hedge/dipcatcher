"""sl(2) structure constants: [e,f]=h, [h,e]=2e, [h,f]=-2f (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

E = np.array([[0.0, 1.0], [0.0, 0.0]])
F = np.array([[0.0, 0.0], [1.0, 0.0]])
H = np.array([[1.0, 0.0], [0.0, -1.0]])


def brac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(x @ y - y @ x)


def _bench_sl2_structure(seed: int = 0) -> float:
    checks = []
    checks.append(np.allclose(brac(E, F), H))
    checks.append(np.allclose(brac(H, E), 2.0 * E))
    checks.append(np.allclose(brac(H, F), -2.0 * F))
    # all traceless
    checks.append(all(np.isclose(np.trace(m), 0.0) for m in (E, F, H)))
    # Jacobi identity
    for x in (E, F, H):
        for y in (E, F, H):
            for z in (E, F, H):
                checks.append(
                    np.allclose(
                        brac(x, brac(y, z)) + brac(y, brac(z, x)) + brac(z, brac(x, y)),
                        np.zeros((2, 2)),
                    )
                )
    # Casimir C = ef + fe + h^2/2 = 3/2 I on defining rep
    c = E @ F + F @ E + 0.5 * H @ H
    checks.append(np.allclose(c, 1.5 * np.eye(2)))
    return float(sum(checks) / len(checks))


def bench_sl2_structure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sl2_structure": _bench_sl2_structure(seed)}
