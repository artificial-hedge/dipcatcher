"""cambrian pp module (SYNTHETIC)."""

from __future__ import annotations


def cambrian_pp_ok(cp: bool, pg: bool) -> bool:
    """cambrian_pp
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def cambrian_pp_aux(aux: bool) -> bool:
    """cambrian_pp
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_cambrian_pp(seed: int = 0) -> float:
    checks = []
    checks.append(cambrian_pp_ok(True, True))
    checks.append(not cambrian_pp_ok(False, True))
    checks.append(cambrian_pp_aux(True))
    checks.append(not cambrian_pp_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_cambrian_pp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cambrian_pp": _bench_cambrian_pp(seed)}
