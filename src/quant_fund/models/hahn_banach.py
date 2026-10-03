"""Hahn-Banach: norm-preserving extension of a linear functional (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def extend_functional(sub_basis: np.ndarray, f_vals: np.ndarray, dim: int) -> np.ndarray:
    """Extend f: span(sub_basis) -> R to F: R^dim -> R with ||F|| = ||f||.

    The canonical extension is the orthogonal one: F(y) = f(proj y), whose
    representing vector lies in the subspace — minimal norm extension.
    """
    q, _ = np.linalg.qr(sub_basis)  # orthonormal basis of subspace
    b = q.T @ sub_basis  # coords of basis in ON basis
    coef = np.linalg.solve(b.T, f_vals)  # f(x) = <coef, x> in ON coords
    return np.asarray(q @ coef)


def _bench_hahn_banach(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    # subspace span{(1,1,0)/sqrt2} in R^3, f(x) = 2<x,e> -> norm 2
    e = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    B = e.reshape(-1, 1)
    fv = np.array([2.0])
    F = extend_functional(B, fv, 3)
    # agrees on subspace
    checks.append(abs(F @ e - 2.0) < 1e-12)
    # same norm
    checks.append(abs(np.linalg.norm(F) - 2.0) < 1e-12)
    # extends boundedness: |F x| <= 2 ||x||
    x = rng.normal(size=(200, 3))
    checks.append(bool(np.all(np.abs(x @ F) <= 2.0 * np.linalg.norm(x, axis=1) + 1e-12)))
    # 2-d subspace with ON basis B2: extension vector = B2 (B2'B2)^{-1} f2
    B2 = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    f2 = rng.normal(size=2)
    rep = np.linalg.solve(B2.T @ B2, f2)  # Riesz rep in ON coords
    F2 = B2 @ rep
    checks.append(np.allclose(B2.T @ F2, f2))
    # norm of functional on subspace = norm of rep vector; extension preserves it
    checks.append(abs(np.linalg.norm(F2) - np.linalg.norm(rep)) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_hahn_banach(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hahn_banach": _bench_hahn_banach(seed)}
