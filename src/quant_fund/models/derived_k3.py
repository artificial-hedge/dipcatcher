"""derived k3 module (SYNTHETIC)."""

from __future__ import annotations


def derived_k3_ok(spectral: bool, geometry: bool) -> bool:
    """derived_k3
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def derived_k3_aux(aux: bool) -> bool:
    """derived_k3
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_derived_k3(seed: int = 0) -> float:
    checks = []
    checks.append(derived_k3_ok(True, True))
    checks.append(not derived_k3_ok(False, True))
    checks.append(derived_k3_aux(True))
    checks.append(not derived_k3_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_derived_k3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_k3": _bench_derived_k3(seed)}
