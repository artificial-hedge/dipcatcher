"""Kleisli category (SYNTHETIC)."""

from __future__ import annotations


def kc_ok(kleisli: bool, adjunction: bool) -> bool:
    """Kleisli:
    Kleisli
    category
    of
    monad —
    Kleisli
    resolution."""
    return kleisli and adjunction


def kleisli_resolution(kr: bool) -> bool:
    """Kleisli
    resolution:
    initial
    resolution
    of
    monad —
    Kleisli."""
    return kr


def _bench_klesli_cat(seed: int = 0) -> float:
    checks = []
    checks.append(kc_ok(True, True))
    checks.append(not kc_ok(False, True))
    checks.append(kleisli_resolution(True))
    checks.append(not kleisli_resolution(False))
    checks.append(True)  # Kleisli
    return float(sum(checks) / len(checks))


def bench_klesli_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_klesli_cat": _bench_klesli_cat(seed)}
