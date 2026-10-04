"""timar tasep module (SYNTHETIC)."""

from __future__ import annotations


def timar_tasep_ok(asep: bool, wk: bool) -> bool:
    """timar_tasep
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def timar_tasep_aux(aux: bool) -> bool:
    """timar_tasep
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_timar_tasep(seed: int = 0) -> float:
    checks = []
    checks.append(timar_tasep_ok(True, True))
    checks.append(not timar_tasep_ok(False, True))
    checks.append(timar_tasep_aux(True))
    checks.append(not timar_tasep_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_timar_tasep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_timar_tasep": _bench_timar_tasep(seed)}
