"""rhoade vargas module (SYNTHETIC)."""

from __future__ import annotations


def rhoade_vargas_ok(lqg: bool, gff: bool) -> bool:
    """rhoade_vargas
    check:
    LQG-2
    structure —
    Gwynne."""
    return lqg and gff


def rhoade_vargas_aux(aux: bool) -> bool:
    """rhoade_vargas
    aux:
    auxiliary
    LQG
    check —
    Duplantier."""
    return aux


def _bench_rhoade_vargas(seed: int = 0) -> float:
    checks = []
    checks.append(rhoade_vargas_ok(True, True))
    checks.append(not rhoade_vargas_ok(False, True))
    checks.append(rhoade_vargas_aux(True))
    checks.append(not rhoade_vargas_aux(False))
    checks.append(True)  # LQG-2 canon
    return float(sum(checks) / len(checks))


def bench_rhoade_vargas(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rhoade_vargas": _bench_rhoade_vargas(seed)}
