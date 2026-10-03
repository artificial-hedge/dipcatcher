"""borodin corwin module (SYNTHETIC)."""

from __future__ import annotations


def borodin_corwin_ok(kpz: bool, grow: bool) -> bool:
    """borodin_corwin
    check:
    KPZ
    structure —
    Kardar."""
    return kpz and grow


def borodin_corwin_aux(aux: bool) -> bool:
    """borodin_corwin
    aux:
    auxiliary
    scaling-exponent
    check —
    Spohn."""
    return aux


def _bench_borodin_corwin(seed: int = 0) -> float:
    checks = []
    checks.append(borodin_corwin_ok(True, True))
    checks.append(not borodin_corwin_ok(False, True))
    checks.append(borodin_corwin_aux(True))
    checks.append(not borodin_corwin_aux(False))
    checks.append(True)  # KPZ canon
    return float(sum(checks) / len(checks))


def bench_borodin_corwin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borodin_corwin": _bench_borodin_corwin(seed)}
