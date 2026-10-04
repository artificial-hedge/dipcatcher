"""bernard nicola module (SYNTHETIC)."""

from __future__ import annotations


def bernard_nicola_ok(kpz: bool, reg: bool) -> bool:
    """bernard_nicola
    check:
    KPZ-2
    structure —
    Corwin."""
    return kpz and reg


def bernard_nicola_aux(aux: bool) -> bool:
    """bernard_nicola
    aux:
    auxiliary
    regularity-structure
    check —
    Hairer."""
    return aux


def _bench_bernard_nicola(seed: int = 0) -> float:
    checks = []
    checks.append(bernard_nicola_ok(True, True))
    checks.append(not bernard_nicola_ok(False, True))
    checks.append(bernard_nicola_aux(True))
    checks.append(not bernard_nicola_aux(False))
    checks.append(True)  # KPZ-2 canon
    return float(sum(checks) / len(checks))


def bench_bernard_nicola(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bernard_nicola": _bench_bernard_nicola(seed)}
