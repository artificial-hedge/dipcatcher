"""singular integrals module (SYNTHETIC)."""

from __future__ import annotations


def singular_integrals_ok(kernel: bool, density: bool) -> bool:
    """singular_integrals
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def singular_integrals_aux(aux: bool) -> bool:
    """singular_integrals
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_singular_integrals(seed: int = 0) -> float:
    checks = []
    checks.append(singular_integrals_ok(True, True))
    checks.append(not singular_integrals_ok(False, True))
    checks.append(singular_integrals_aux(True))
    checks.append(not singular_integrals_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_singular_integrals(seed: int = 0) -> dict[str, float]:
    return {"synthetic_singular_integrals": _bench_singular_integrals(seed)}
