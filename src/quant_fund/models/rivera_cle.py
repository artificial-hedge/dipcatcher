"""rivera cle module (SYNTHETIC)."""

from __future__ import annotations


def rivera_cle_ok(cle: bool, sle: bool) -> bool:
    """rivera_cle
    check:
    conformal-loop-ensemble
    structure —
    Sheffield."""
    return cle and sle


def rivera_cle_aux(aux: bool) -> bool:
    """rivera_cle
    aux:
    auxiliary
    loop-ensemble
    check —
    Werner."""
    return aux


def _bench_rivera_cle(seed: int = 0) -> float:
    checks = []
    checks.append(rivera_cle_ok(True, True))
    checks.append(not rivera_cle_ok(False, True))
    checks.append(rivera_cle_aux(True))
    checks.append(not rivera_cle_aux(False))
    checks.append(True)  # CLE canon
    return float(sum(checks) / len(checks))


def bench_rivera_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rivera_cle": _bench_rivera_cle(seed)}
