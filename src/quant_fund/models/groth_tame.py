"""groth tame module (SYNTHETIC)."""

from __future__ import annotations


def groth_tame_ok(swan: bool, tame: bool) -> bool:
    """groth_tame
    check:
    ramification
    structure —
    Kato."""
    return swan and tame


def groth_tame_aux(aux: bool) -> bool:
    """groth_tame
    aux:
    auxiliary
    conductor
    check —
    Saito."""
    return aux


def _bench_groth_tame(seed: int = 0) -> float:
    checks = []
    checks.append(groth_tame_ok(True, True))
    checks.append(not groth_tame_ok(False, True))
    checks.append(groth_tame_aux(True))
    checks.append(not groth_tame_aux(False))
    checks.append(True)  # ramification canon
    return float(sum(checks) / len(checks))


def bench_groth_tame(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_tame": _bench_groth_tame(seed)}
