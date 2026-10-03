"""Prony's method: fit sum-of-exponentials to a sequence."""

import numpy as np

_SEED = 20261231 + 749


def prony_fit(y: np.ndarray, p: int) -> np.ndarray:
    """Solve for AR coefficients a s.t. y[n] = -sum a_k y[n-k]."""
    n = len(y)
    A = np.stack([y[p - 1 - k : n - 1 - k] for k in range(p)], axis=1)
    b = -y[p:]
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    out: np.ndarray = a
    return out


def bench_prony_model(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # y = c1*r1^n + c2*r2^n (exact 2-exponential)
    r1, r2 = 0.8, 0.5
    n = np.arange(30)
    y = 2.0 * r1**n + 1.0 * r2**n + 1e-6 * rng.normal(size=30)
    a = prony_fit(y, 2)
    # characteristic roots should recover r1, r2
    roots = np.roots(np.concatenate([[1.0], a]))
    got = sorted(np.abs(roots), reverse=True)
    expect = sorted([r1, r2], reverse=True)
    err = float(max(abs(g - e) for g, e in zip(got, expect, strict=True)))
    return {"synthetic_prony_roots": float(err < 0.05)}
