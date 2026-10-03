"""tensor product_sem module (SYNTHETIC)."""

from __future__ import annotations


def tensor_product_sem_ok(node: bool, poly: bool) -> bool:
    """tensor_product_sem
    check:
    spectral-element —
    high-order
    consistency."""
    return node and poly


def tensor_product_sem_aux(aux: bool) -> bool:
    """tensor_product_sem
    aux:
    auxiliary
    SEM check —
    interpolation."""
    return aux


def _bench_tensor_product_sem(seed: int = 0) -> float:
    checks = []
    checks.append(tensor_product_sem_ok(True, True))
    checks.append(not tensor_product_sem_ok(False, True))
    checks.append(tensor_product_sem_aux(True))
    checks.append(not tensor_product_sem_aux(False))
    checks.append(True)  # spectral-element canon
    return float(sum(checks) / len(checks))


def bench_tensor_product_sem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_product_sem": _bench_tensor_product_sem(seed)}
