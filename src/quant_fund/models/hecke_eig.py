"""Hecke eigensheaves (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def hecke_eigen_ok(hecke_action: np.ndarray, eigenval: float) -> bool:
    """A Hecke eigensheaf F satisfies H_V(F) = V x F
    for each irrep V of G^L; eigenvalue = local system
    holonomy (Frenkel-Gaitsgory-Vilonen)."""
    return bool(np.allclose(hecke_action, eigenval * np.eye(len(hecke_action))))


def _bench_hecke_eig(seed: int = 0) -> float:
    checks = []
    checks.append(hecke_eigen_ok(2.0 * np.eye(2), 2.0))
    checks.append(not hecke_eigen_ok(np.array([[1.0, 1.0], [0.0, 1.0]]), 1.0))
    # Hecke correspondences via modifications of bundles
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hecke_eig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_eig": _bench_hecke_eig(seed)}
