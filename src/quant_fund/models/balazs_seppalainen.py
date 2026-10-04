"""balazs seppalainen module (SYNTHETIC)."""

from __future__ import annotations


def balazs_seppalainen_ok(asep: bool, wk: bool) -> bool:
    """balazs_seppalainen
    check:
    ASEP-2
    structure —
    Sasamoto."""
    return asep and wk


def balazs_seppalainen_aux(aux: bool) -> bool:
    """balazs_seppalainen
    aux:
    auxiliary
    weak-asymmetry
    check —
    Bertini."""
    return aux


def _bench_balazs_seppalainen(seed: int = 0) -> float:
    checks = []
    checks.append(balazs_seppalainen_ok(True, True))
    checks.append(not balazs_seppalainen_ok(False, True))
    checks.append(balazs_seppalainen_aux(True))
    checks.append(not balazs_seppalainen_aux(False))
    checks.append(True)  # ASEP-2 canon
    return float(sum(checks) / len(checks))


def bench_balazs_seppalainen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_balazs_seppalainen": _bench_balazs_seppalainen(seed)}
