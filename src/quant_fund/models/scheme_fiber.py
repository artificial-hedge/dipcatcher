"""Fiber product of affine schemes via tensor product (SYNTHETIC)."""

from __future__ import annotations


def tensor_dim(a: int, b: int) -> int:
    """Spec(A x_k B) = Spec A x_{Spec k} Spec B; dim of tensor
    of finite k-algebras multiplies."""
    return a * b


def _bench_scheme_fiber(seed: int = 0) -> float:
    checks = []
    # A1 x A1 = A2
    checks.append(tensor_dim(1, 1) == 1)
    # k[x]/(x^2) x k[y]/(y^3): dim 2*3 = 6
    checks.append(tensor_dim(2, 3) == 6)
    # base change preserves dimension of finite algebras
    checks.append(tensor_dim(5, 1) == 5)
    # fiber over a point: A x_k k = A
    checks.append(tensor_dim(4, 1) == 4)
    # empty fiber: disjoint specs give product zero
    checks.append(tensor_dim(0, 3) == 0)
    return float(sum(checks) / len(checks))


def bench_scheme_fiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scheme_fiber": _bench_scheme_fiber(seed)}
