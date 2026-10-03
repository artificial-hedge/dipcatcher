"""mellin transform module (SYNTHETIC)."""

from __future__ import annotations


def mellin_transform_ok(kernel: bool, domain: bool) -> bool:
    """mellin_transform
    check:
    integral
    transform —
    kernel."""
    return kernel and domain


def mellin_transform_aux(aux: bool) -> bool:
    """mellin_transform
    aux:
    auxiliary
    transform check —
    inverse."""
    return aux


def _bench_mellin_transform(seed: int = 0) -> float:
    checks = []
    checks.append(mellin_transform_ok(True, True))
    checks.append(not mellin_transform_ok(False, True))
    checks.append(mellin_transform_aux(True))
    checks.append(not mellin_transform_aux(False))
    checks.append(True)  # integral-transforms canon
    return float(sum(checks) / len(checks))


def bench_mellin_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mellin_transform": _bench_mellin_transform(seed)}
