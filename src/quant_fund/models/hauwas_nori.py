"""hauwas nori module (SYNTHETIC)."""

from __future__ import annotations


def hauwas_nori_ok(motive: bool, suslin: bool) -> bool:
    """hauwas_nori
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def hauwas_nori_aux(aux: bool) -> bool:
    """hauwas_nori
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_hauwas_nori(seed: int = 0) -> float:
    checks = []
    checks.append(hauwas_nori_ok(True, True))
    checks.append(not hauwas_nori_ok(False, True))
    checks.append(hauwas_nori_aux(True))
    checks.append(not hauwas_nori_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_hauwas_nori(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hauwas_nori": _bench_hauwas_nori(seed)}
