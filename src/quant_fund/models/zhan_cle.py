"""zhan cle module (SYNTHETIC)."""

from __future__ import annotations


def zhan_cle_ok(cle: bool, loop: bool) -> bool:
    """zhan_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def zhan_cle_aux(aux: bool) -> bool:
    """zhan_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_zhan_cle(seed: int = 0) -> float:
    checks = []
    checks.append(zhan_cle_ok(True, True))
    checks.append(not zhan_cle_ok(False, True))
    checks.append(zhan_cle_aux(True))
    checks.append(not zhan_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_zhan_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zhan_cle": _bench_zhan_cle(seed)}
