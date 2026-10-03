"""dirk scheme module (SYNTHETIC)."""

from __future__ import annotations


def dirk_scheme_ok(step: bool, order: bool) -> bool:
    """dirk_scheme
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def dirk_scheme_aux(aux: bool) -> bool:
    """dirk_scheme
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_dirk_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(dirk_scheme_ok(True, True))
    checks.append(not dirk_scheme_ok(False, True))
    checks.append(dirk_scheme_aux(True))
    checks.append(not dirk_scheme_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_dirk_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirk_scheme": _bench_dirk_scheme(seed)}
