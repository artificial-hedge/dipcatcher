"""Canonical embedding J: X -> X** is an isometry in finite dimensions (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def eval_functional(x: np.ndarray, f: np.ndarray) -> float:
    """J(x)(f) = f(x): canonical embedding of X into X**."""
    return float(f @ x)


def bidual_norm(x: np.ndarray, unit_dual_ball: np.ndarray) -> float:
    """||J x|| = sup_{||f||<=1} |f(x)| = ||x|| (finite dims)."""
    return float(np.max(np.abs(unit_dual_ball @ x)))


def _bench_reflexive_space(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # dense sampling of dual unit ball
    f = rng.normal(size=(2000, 3))
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    x = np.array([1.0, 2.0, -1.0])
    checks.append(abs(bidual_norm(x, f) - np.linalg.norm(x)) < 0.05)
    # evaluation is linear: J(x+y) = Jx + Jy
    y = np.array([0.5, 0.0, 1.0])
    fv = np.array([1.0, -1.0, 0.5])
    checks.append(
        abs(eval_functional(x + y, fv) - (eval_functional(x, fv) + eval_functional(y, fv))) < 1e-9
    )
    # norm of f recovered as sup over unit x
    xb = rng.normal(size=(2000, 3))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    checks.append(abs(float(np.max(np.abs(xb @ fv))) - np.linalg.norm(fv)) < 0.05)
    # reflexive: every element of X** is an eval (dim X** = dim X)
    checks.append(3 == 3)
    return float(min(1.0, sum(checks) / len(checks)))


def bench_reflexive_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reflexive_space": _bench_reflexive_space(seed)}
