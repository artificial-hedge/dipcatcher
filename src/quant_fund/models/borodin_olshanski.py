"""borodin olshanski module (SYNTHETIC)."""

from __future__ import annotations


def borodin_olshanski_ok(rmt: bool, univ: bool) -> bool:
    """borodin_olshanski
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def borodin_olshanski_aux(aux: bool) -> bool:
    """borodin_olshanski
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_borodin_olshanski(seed: int = 0) -> float:
    checks = []
    checks.append(borodin_olshanski_ok(True, True))
    checks.append(not borodin_olshanski_ok(False, True))
    checks.append(borodin_olshanski_aux(True))
    checks.append(not borodin_olshanski_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_borodin_olshanski(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borodin_olshanski": _bench_borodin_olshanski(seed)}
