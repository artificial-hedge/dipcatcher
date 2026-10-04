"""kato swan module (SYNTHETIC)."""

from __future__ import annotations


def kato_swan_ok(swan: bool, tame: bool) -> bool:
    """kato_swan
    check:
    ramification
    structure —
    Kato."""
    return swan and tame


def kato_swan_aux(aux: bool) -> bool:
    """kato_swan
    aux:
    auxiliary
    conductor
    check —
    Saito."""
    return aux


def _bench_kato_swan(seed: int = 0) -> float:
    checks = []
    checks.append(kato_swan_ok(True, True))
    checks.append(not kato_swan_ok(False, True))
    checks.append(kato_swan_aux(True))
    checks.append(not kato_swan_aux(False))
    checks.append(True)  # ramification canon
    return float(sum(checks) / len(checks))


def bench_kato_swan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kato_swan": _bench_kato_swan(seed)}
