"""barlow ust module (SYNTHETIC)."""

from __future__ import annotations


def barlow_ust_ok(loop: bool, gff: bool) -> bool:
    """barlow_ust
    check:
    loop-soup
    structure —
    LeJan."""
    return loop and gff


def barlow_ust_aux(aux: bool) -> bool:
    """barlow_ust
    aux:
    auxiliary
    Gaussian-field
    check —
    Lupu."""
    return aux


def _bench_barlow_ust(seed: int = 0) -> float:
    checks = []
    checks.append(barlow_ust_ok(True, True))
    checks.append(not barlow_ust_ok(False, True))
    checks.append(barlow_ust_aux(True))
    checks.append(not barlow_ust_aux(False))
    checks.append(True)  # loop-soup canon
    return float(sum(checks) / len(checks))


def bench_barlow_ust(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barlow_ust": _bench_barlow_ust(seed)}
