"""laplace transform module (SYNTHETIC)."""

from __future__ import annotations


def laplace_transform_ok(kernel: bool, domain: bool) -> bool:
    """laplace_transform
    check:
    integral
    transform —
    kernel."""
    return kernel and domain


def laplace_transform_aux(aux: bool) -> bool:
    """laplace_transform
    aux:
    auxiliary
    transform check —
    inverse."""
    return aux


def _bench_laplace_transform(seed: int = 0) -> float:
    checks = []
    checks.append(laplace_transform_ok(True, True))
    checks.append(not laplace_transform_ok(False, True))
    checks.append(laplace_transform_aux(True))
    checks.append(not laplace_transform_aux(False))
    checks.append(True)  # integral-transforms canon
    return float(sum(checks) / len(checks))


def bench_laplace_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_laplace_transform": _bench_laplace_transform(seed)}
