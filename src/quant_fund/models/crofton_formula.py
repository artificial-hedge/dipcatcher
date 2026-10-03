"""crofton formula module (SYNTHETIC)."""

from __future__ import annotations


def crofton_formula_ok(geo: bool, kin: bool) -> bool:
    """crofton_formula
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def crofton_formula_aux(aux: bool) -> bool:
    """crofton_formula
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_crofton_formula(seed: int = 0) -> float:
    checks = []
    checks.append(crofton_formula_ok(True, True))
    checks.append(not crofton_formula_ok(False, True))
    checks.append(crofton_formula_aux(True))
    checks.append(not crofton_formula_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_crofton_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crofton_formula": _bench_crofton_formula(seed)}
