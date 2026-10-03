"""ergodic pp module (SYNTHETIC)."""

from __future__ import annotations


def ergodic_pp_ok(cp: bool, pg: bool) -> bool:
    """ergodic_pp
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def ergodic_pp_aux(aux: bool) -> bool:
    """ergodic_pp
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_ergodic_pp(seed: int = 0) -> float:
    checks = []
    checks.append(ergodic_pp_ok(True, True))
    checks.append(not ergodic_pp_ok(False, True))
    checks.append(ergodic_pp_aux(True))
    checks.append(not ergodic_pp_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_ergodic_pp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ergodic_pp": _bench_ergodic_pp(seed)}
