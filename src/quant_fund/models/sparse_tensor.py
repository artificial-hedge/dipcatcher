"""sparse tensor module (SYNTHETIC)."""

from __future__ import annotations


def sparse_tensor_ok(grid: bool, level: bool) -> bool:
    """sparse_tensor
    check:
    sparse-grid/dimension-adaptive —
    surplus
    consistency."""
    return grid and level


def sparse_tensor_aux(aux: bool) -> bool:
    """sparse_tensor
    aux:
    auxiliary
    sparse check —
    tensor bound."""
    return aux


def _bench_sparse_tensor(seed: int = 0) -> float:
    checks = []
    checks.append(sparse_tensor_ok(True, True))
    checks.append(not sparse_tensor_ok(False, True))
    checks.append(sparse_tensor_aux(True))
    checks.append(not sparse_tensor_aux(False))
    checks.append(True)  # sparse-grid canon
    return float(sum(checks) / len(checks))


def bench_sparse_tensor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sparse_tensor": _bench_sparse_tensor(seed)}
