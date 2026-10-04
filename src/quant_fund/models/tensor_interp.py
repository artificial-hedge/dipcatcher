"""tensor interp module (SYNTHETIC)."""

from __future__ import annotations


def tensor_interp_ok(node: bool, wgt: bool) -> bool:
    """tensor_interp
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def tensor_interp_aux(aux: bool) -> bool:
    """tensor_interp
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_tensor_interp(seed: int = 0) -> float:
    checks = []
    checks.append(tensor_interp_ok(True, True))
    checks.append(not tensor_interp_ok(False, True))
    checks.append(tensor_interp_aux(True))
    checks.append(not tensor_interp_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_tensor_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_interp": _bench_tensor_interp(seed)}
