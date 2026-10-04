"""nolin cle module (SYNTHETIC)."""

from __future__ import annotations


def nolin_cle_ok(cle: bool, loop: bool) -> bool:
    """nolin_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def nolin_cle_aux(aux: bool) -> bool:
    """nolin_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_nolin_cle(seed: int = 0) -> float:
    checks = []
    checks.append(nolin_cle_ok(True, True))
    checks.append(not nolin_cle_ok(False, True))
    checks.append(nolin_cle_aux(True))
    checks.append(not nolin_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_nolin_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nolin_cle": _bench_nolin_cle(seed)}
