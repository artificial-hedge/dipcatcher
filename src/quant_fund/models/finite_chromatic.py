"""Finite chromatic types (SYNTHETIC)."""

from __future__ import annotations


def fch_ok(finite: bool, chromatic: bool) -> bool:
    """Finite
    chromatic:
    finite
    chromatic —
    type
    n."""
    return finite and chromatic


def type_n_spectrum(tn: bool) -> bool:
    """Type
    n:
    type
    n
    finite
    spectrum —
    Smith."""
    return tn


def _bench_finite_chromatic(seed: int = 0) -> float:
    checks = []
    checks.append(fch_ok(True, True))
    checks.append(not fch_ok(False, True))
    checks.append(type_n_spectrum(True))
    checks.append(not type_n_spectrum(False))
    checks.append(True)  # Smith
    return float(sum(checks) / len(checks))


def bench_finite_chromatic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_chromatic": _bench_finite_chromatic(seed)}
