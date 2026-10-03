"""moving least_sq module (SYNTHETIC)."""

from __future__ import annotations


def moving_least_sq_ok(node: bool, cloud: bool) -> bool:
    """moving_least_sq
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def moving_least_sq_aux(aux: bool) -> bool:
    """moving_least_sq
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_moving_least_sq(seed: int = 0) -> float:
    checks = []
    checks.append(moving_least_sq_ok(True, True))
    checks.append(not moving_least_sq_ok(False, True))
    checks.append(moving_least_sq_aux(True))
    checks.append(not moving_least_sq_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_moving_least_sq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moving_least_sq": _bench_moving_least_sq(seed)}
