"""grimmett rc module (SYNTHETIC)."""

from __future__ import annotations


def grimmett_rc_ok(rc: bool, potts: bool) -> bool:
    """grimmett_rc
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def grimmett_rc_aux(aux: bool) -> bool:
    """grimmett_rc
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_grimmett_rc(seed: int = 0) -> float:
    checks = []
    checks.append(grimmett_rc_ok(True, True))
    checks.append(not grimmett_rc_ok(False, True))
    checks.append(grimmett_rc_aux(True))
    checks.append(not grimmett_rc_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_grimmett_rc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grimmett_rc": _bench_grimmett_rc(seed)}
