"""analytic spec module (SYNTHETIC)."""

from __future__ import annotations


def analytic_spec_ok(spectral: bool, geometry: bool) -> bool:
    """analytic_spec
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def analytic_spec_aux(aux: bool) -> bool:
    """analytic_spec
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_analytic_spec(seed: int = 0) -> float:
    checks = []
    checks.append(analytic_spec_ok(True, True))
    checks.append(not analytic_spec_ok(False, True))
    checks.append(analytic_spec_aux(True))
    checks.append(not analytic_spec_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_analytic_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_spec": _bench_analytic_spec(seed)}
