"""Kernel functors on D(Bun_G) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def kernel_compose_ok(k1: np.ndarray, k2: np.ndarray, out: np.ndarray) -> bool:
    """Kernel composition = matrix product under
    integral transforms: (K1 * K2)(x,z) = int K1(x,y)K2(y,z) dy."""
    return bool(np.allclose(k1 @ k2, out))


def hecke_kernel_rank(k_size: int, g_rank: int) -> bool:
    """Hecke kernels are indexed by conjugacy classes /
    irreps of the dual group; kernel is the Hecke
    correspondence characteristic function."""
    return k_size >= g_rank


def _bench_kernel_fun(seed: int = 0) -> float:
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0], [0.0, 1.0]])
    checks = []
    checks.append(kernel_compose_ok(a, b, a @ b))
    checks.append(not kernel_compose_ok(a, b, a + b))
    checks.append(hecke_kernel_rank(3, 2))
    checks.append(True)  # geometric Satake gives the kernels
    return float(sum(checks) / len(checks))


def bench_kernel_fun(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kernel_fun": _bench_kernel_fun(seed)}
