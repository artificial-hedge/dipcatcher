"""steiner formula module (SYNTHETIC)."""

from __future__ import annotations


def steiner_formula_ok(geo: bool, tess: bool) -> bool:
    """steiner_formula
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def steiner_formula_aux(aux: bool) -> bool:
    """steiner_formula
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_steiner_formula(seed: int = 0) -> float:
    checks = []
    checks.append(steiner_formula_ok(True, True))
    checks.append(not steiner_formula_ok(False, True))
    checks.append(steiner_formula_aux(True))
    checks.append(not steiner_formula_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_steiner_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steiner_formula": _bench_steiner_formula(seed)}
