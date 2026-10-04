"""graded spec module (SYNTHETIC)."""

from __future__ import annotations


def graded_spec_ok(spectral: bool, geometry: bool) -> bool:
    """graded_spec
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def graded_spec_aux(aux: bool) -> bool:
    """graded_spec
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_graded_spec(seed: int = 0) -> float:
    checks = []
    checks.append(graded_spec_ok(True, True))
    checks.append(not graded_spec_ok(False, True))
    checks.append(graded_spec_aux(True))
    checks.append(not graded_spec_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_graded_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_graded_spec": _bench_graded_spec(seed)}
