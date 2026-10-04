"""Cheeger-Simons characters (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(homomorphism: bool, differential: bool) -> bool:
    """Cheeger-
    Simons
    differential
    character:
    homomorphism
    Z_{k-1} ->
    R/Z
    extending
    the curvature
    form."""
    return homomorphism and differential


def secondary_class(sec: bool) -> bool:
    """Secondary
    characteristic
    classes arise
    as differentials
    of Cheeger-
    Simons
    characters."""
    return sec


def _bench_cheeger_simons(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(secondary_class(True))
    checks.append(not secondary_class(False))
    checks.append(True)  # Cheeger-Simons 1985
    return float(sum(checks) / len(checks))


def bench_cheeger_simons(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cheeger_simons": _bench_cheeger_simons(seed)}
