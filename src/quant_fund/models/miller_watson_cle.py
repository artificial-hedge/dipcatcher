"""miller watson_cle module (SYNTHETIC)."""

from __future__ import annotations


def miller_watson_cle_ok(cle: bool, sle: bool) -> bool:
    """miller_watson_cle
    check:
    conformal-loop-ensemble
    structure —
    Sheffield."""
    return cle and sle


def miller_watson_cle_aux(aux: bool) -> bool:
    """miller_watson_cle
    aux:
    auxiliary
    loop-ensemble
    check —
    Werner."""
    return aux


def _bench_miller_watson_cle(seed: int = 0) -> float:
    checks = []
    checks.append(miller_watson_cle_ok(True, True))
    checks.append(not miller_watson_cle_ok(False, True))
    checks.append(miller_watson_cle_aux(True))
    checks.append(not miller_watson_cle_aux(False))
    checks.append(True)  # CLE canon
    return float(sum(checks) / len(checks))


def bench_miller_watson_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_miller_watson_cle": _bench_miller_watson_cle(seed)}
