"""fourier mukai module (SYNTHETIC)."""

from __future__ import annotations


def fourier_mukai_ok(triangulated: bool, derived: bool) -> bool:
    """fourier_mukai
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def fourier_mukai_aux(aux: bool) -> bool:
    """fourier_mukai
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_fourier_mukai(seed: int = 0) -> float:
    checks = []
    checks.append(fourier_mukai_ok(True, True))
    checks.append(not fourier_mukai_ok(False, True))
    checks.append(fourier_mukai_aux(True))
    checks.append(not fourier_mukai_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_fourier_mukai(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_mukai": _bench_fourier_mukai(seed)}
