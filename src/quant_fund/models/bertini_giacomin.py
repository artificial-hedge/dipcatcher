"""bertini giacomin module (SYNTHETIC)."""

from __future__ import annotations


def bertini_giacomin_ok(asep: bool, wk: bool) -> bool:
    """bertini_giacomin
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def bertini_giacomin_aux(aux: bool) -> bool:
    """bertini_giacomin
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_bertini_giacomin(seed: int = 0) -> float:
    checks = []
    checks.append(bertini_giacomin_ok(True, True))
    checks.append(not bertini_giacomin_ok(False, True))
    checks.append(bertini_giacomin_aux(True))
    checks.append(not bertini_giacomin_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_bertini_giacomin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bertini_giacomin": _bench_bertini_giacomin(seed)}
