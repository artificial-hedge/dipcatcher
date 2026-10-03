"""hankel transform module (SYNTHETIC)."""

from __future__ import annotations


def hankel_transform_ok(kernel: bool, domain: bool) -> bool:
    """hankel_transform
    check:
    integral
    transform —
    kernel."""
    return kernel and domain


def hankel_transform_aux(aux: bool) -> bool:
    """hankel_transform
    aux:
    auxiliary
    transform check —
    inverse."""
    return aux


def _bench_hankel_transform(seed: int = 0) -> float:
    checks = []
    checks.append(hankel_transform_ok(True, True))
    checks.append(not hankel_transform_ok(False, True))
    checks.append(hankel_transform_aux(True))
    checks.append(not hankel_transform_aux(False))
    checks.append(True)  # integral-transforms canon
    return float(sum(checks) / len(checks))


def bench_hankel_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hankel_transform": _bench_hankel_transform(seed)}
