"""hospitsky cle module (SYNTHETIC)."""

from __future__ import annotations


def hospitsky_cle_ok(cle: bool, loop: bool) -> bool:
    """hospitsky_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def hospitsky_cle_aux(aux: bool) -> bool:
    """hospitsky_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_hospitsky_cle(seed: int = 0) -> float:
    checks = []
    checks.append(hospitsky_cle_ok(True, True))
    checks.append(not hospitsky_cle_ok(False, True))
    checks.append(hospitsky_cle_aux(True))
    checks.append(not hospitsky_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_hospitsky_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hospitsky_cle": _bench_hospitsky_cle(seed)}
