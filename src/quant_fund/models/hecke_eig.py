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
    # a diagonal-but-non-scalar action is not an eigensheaf either
    checks.append(not hecke_eigen_ok(np.diag([1.0, 2.0, 3.0]), 1.0))
    # scaled identity in any dimension is
    checks.append(hecke_eigen_ok(-1.5 * np.eye(4), -1.5))
    return float(sum(checks) / len(checks))


def bench_hecke_eig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hecke_eig": _bench_hecke_eig(seed)}
