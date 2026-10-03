"""Compact operators on finite dim: finite-rank approximation property (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def finite_rank_approx(a: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approx via SVD truncation."""
    u, s, vt = np.linalg.svd(a)
    s2 = s.copy()
    s2[r:] = 0.0
    return np.asarray(u @ np.diag(s2) @ vt)


def op_norm(a: np.ndarray) -> float:
    return float(np.linalg.svd(a, compute_uv=False)[0])


def _bench_compact_operator(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    # diagonal operator with decaying entries ~ compact (limit of finite rank)
    d = np.diag([1.0 / (i + 1) for i in range(6)])
    # approximating error after rank-2 truncation = next singular value = 1/3
    checks.append(abs(op_norm(d - finite_rank_approx(d, 2)) - 1.0 / 3.0) < 1e-9)
    # finite rank ops are "compact": error -> 0 at full rank
    checks.append(op_norm(d - finite_rank_approx(d, 6)) < 1e-12)
    a = rng.standard_normal((5, 5))
    checks.append(op_norm(a - finite_rank_approx(a, 5)) < 1e-9)
    # Eckart-Young: truncated approx <= any other rank-2 approx in op norm
    b = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 5))
    checks.append(op_norm(a - finite_rank_approx(a, 2)) <= op_norm(a - b) + 1e-9)
    # identity on infinite-dim isn't compact — here finite version has full-rank error 1
    checks.append(abs(op_norm(np.eye(4) - np.zeros((4, 4))) - 1.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_compact_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_operator": _bench_compact_operator(seed)}
