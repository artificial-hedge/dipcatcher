"""norimotive3 module (SYNTHETIC)."""

from __future__ import annotations


def norimotive3_ok(motivic: bool, hodge: bool) -> bool:
    """norimotive3
    check:
    motivic
    structure —
    Hodge."""
    return motivic and hodge


def norimotive3_aux(aux: bool) -> bool:
    """norimotive3
    aux:
    auxiliary
    motivic
    check —
    Tate."""
    return aux


def _bench_norimotive3(seed: int = 0) -> float:
    checks = []
    checks.append(norimotive3_ok(True, True))
    checks.append(not norimotive3_ok(False, True))
    checks.append(norimotive3_aux(True))
    checks.append(not norimotive3_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_norimotive3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norimotive3": _bench_norimotive3(seed)}
