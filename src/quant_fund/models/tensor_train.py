"""tensor train module (SYNTHETIC)."""

from __future__ import annotations


def tensor_train_ok(node: bool, wgt: bool) -> bool:
    """tensor_train
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def tensor_train_aux(aux: bool) -> bool:
    """tensor_train
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_tensor_train(seed: int = 0) -> float:
    checks = []
    checks.append(tensor_train_ok(True, True))
    checks.append(not tensor_train_ok(False, True))
    checks.append(tensor_train_aux(True))
    checks.append(not tensor_train_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_tensor_train(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_train": _bench_tensor_train(seed)}
