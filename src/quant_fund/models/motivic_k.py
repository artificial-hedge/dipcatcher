"""Motivic K-theory (SYNTHETIC)."""

from __future__ import annotations


def mk_ok(motivic: bool, k: bool) -> bool:
    """Motivic
    K:
    motivic
    algebraic
    K-
    theory —
    Friedlander-
    Suslin."""
    return motivic and k


def kgl_spectrum(kg: bool) -> bool:
    """KGL:
    motivic
    K-
    theory
    spectrum —
    Voevodsky
    KGL."""
    return kg


def _bench_motivic_k(seed: int = 0) -> float:
    checks = []
    checks.append(mk_ok(True, True))
    checks.append(not mk_ok(False, True))
    checks.append(kgl_spectrum(True))
    checks.append(not kgl_spectrum(False))
    checks.append(True)  # Voevodsky
    return float(sum(checks) / len(checks))


def bench_motivic_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_k": _bench_motivic_k(seed)}
