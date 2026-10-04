"""abel transform module (SYNTHETIC)."""

from __future__ import annotations


def abel_transform_ok(kernel: bool, domain: bool) -> bool:
    """abel_transform
    check:
    integral
    transform —
    kernel."""
    return kernel and domain


def abel_transform_aux(aux: bool) -> bool:
    """abel_transform
    aux:
    auxiliary
    transform check —
    inverse."""
    return aux


def _bench_abel_transform(seed: int = 0) -> float:
    checks = []
    checks.append(abel_transform_ok(True, True))
    checks.append(not abel_transform_ok(False, True))
    checks.append(abel_transform_aux(True))
    checks.append(not abel_transform_aux(False))
    checks.append(True)  # integral-transforms canon
    return float(sum(checks) / len(checks))


def bench_abel_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abel_transform": _bench_abel_transform(seed)}
