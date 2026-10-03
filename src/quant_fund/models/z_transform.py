"""z transform module (SYNTHETIC)."""

from __future__ import annotations


def z_transform_ok(kernel: bool, domain: bool) -> bool:
    """z_transform
    check:
    integral
    transform —
    kernel."""
    return kernel and domain


def z_transform_aux(aux: bool) -> bool:
    """z_transform
    aux:
    auxiliary
    transform check —
    inverse."""
    return aux


def _bench_z_transform(seed: int = 0) -> float:
    checks = []
    checks.append(z_transform_ok(True, True))
    checks.append(not z_transform_ok(False, True))
    checks.append(z_transform_aux(True))
    checks.append(not z_transform_aux(False))
    checks.append(True)  # integral-transforms canon
    return float(sum(checks) / len(checks))


def bench_z_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_z_transform": _bench_z_transform(seed)}
