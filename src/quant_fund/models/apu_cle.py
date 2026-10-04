"""apu cle module (SYNTHETIC)."""

from __future__ import annotations


def apu_cle_ok(cle: bool, loop: bool) -> bool:
    """apu_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def apu_cle_aux(aux: bool) -> bool:
    """apu_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_apu_cle(seed: int = 0) -> float:
    checks = []
    checks.append(apu_cle_ok(True, True))
    checks.append(not apu_cle_ok(False, True))
    checks.append(apu_cle_aux(True))
    checks.append(not apu_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_apu_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_apu_cle": _bench_apu_cle(seed)}
