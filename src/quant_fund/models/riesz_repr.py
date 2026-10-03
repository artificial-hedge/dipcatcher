"""Riesz representation: every f in H* is f(x) = <x, y_f> (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def riesz_vector(f_vals: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Given a functional's values on basis vectors, recover y in H."""
    g_mat = basis @ basis.T
    coef = np.linalg.solve(g_mat, f_vals)
    return np.asarray(basis.T @ coef)


def _bench_riesz_repr(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    y_true = rng.normal(size=6)
    B = np.eye(6)
    fv = B @ y_true
    y = riesz_vector(fv, B)
    checks.append(np.allclose(y, y_true))
    # non-orthonormal basis: solve Gram system
    B2 = rng.normal(size=(6, 4))
    y_true2 = rng.normal(size=6)
    fv2 = B2.T @ y_true2
    y2 = riesz_vector(fv2, B2.T)
    # recovered functional agrees on the basis
    checks.append(np.allclose(B2.T @ y2, fv2))
    # y lies in span of basis (unique minimal representative)
    coef = np.linalg.lstsq(B2, y2, rcond=None)[0]
    checks.append(np.allclose(B2 @ coef, y2))
    # norm of functional = norm of representing vector
    x = rng.normal(size=(500, 4)) @ B2.T
    ratio = np.max(np.abs(x @ y2) / np.linalg.norm(x, axis=1))
    checks.append(ratio <= np.linalg.norm(y2) + 1e-9)
    checks.append(ratio > 0.9 * np.linalg.norm(y2))
    return float(sum(checks) / len(checks))


def bench_riesz_repr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riesz_repr": _bench_riesz_repr(seed)}
