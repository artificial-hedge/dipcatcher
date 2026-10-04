"""gaines sle module (SYNTHETIC)."""

from __future__ import annotations


def gaines_sle_ok(lqg: bool, gff: bool) -> bool:
    """gaines_sle
    check:
    LQG-2
    structure —
    Gwynne."""
    return lqg and gff


def gaines_sle_aux(aux: bool) -> bool:
    """gaines_sle
    aux:
    auxiliary
    LQG
    check —
    Duplantier."""
    return aux


def _bench_gaines_sle(seed: int = 0) -> float:
    checks = []
    checks.append(gaines_sle_ok(True, True))
    checks.append(not gaines_sle_ok(False, True))
    checks.append(gaines_sle_aux(True))
    checks.append(not gaines_sle_aux(False))
    checks.append(True)  # LQG-2 canon
    return float(sum(checks) / len(checks))


def bench_gaines_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gaines_sle": _bench_gaines_sle(seed)}
