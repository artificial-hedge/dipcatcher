"""mls shape module (SYNTHETIC)."""

from __future__ import annotations


def mls_shape_ok(node: bool, cloud: bool) -> bool:
    """mls_shape
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def mls_shape_aux(aux: bool) -> bool:
    """mls_shape
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_mls_shape(seed: int = 0) -> float:
    checks = []
    checks.append(mls_shape_ok(True, True))
    checks.append(not mls_shape_ok(False, True))
    checks.append(mls_shape_aux(True))
    checks.append(not mls_shape_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_mls_shape(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mls_shape": _bench_mls_shape(seed)}
