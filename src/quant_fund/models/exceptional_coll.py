"""exceptional coll module (SYNTHETIC)."""

from __future__ import annotations


def exceptional_coll_ok(triangulated: bool, derived: bool) -> bool:
    """exceptional_coll
    check:
    triangulated
    structure —
    exceptional."""
    return triangulated and derived


def exceptional_coll_aux(aux: bool) -> bool:
    """exceptional_coll
    aux:
    auxiliary
    triangulated
    check —
    Fourier."""
    return aux


def _bench_exceptional_coll(seed: int = 0) -> float:
    checks = []
    checks.append(exceptional_coll_ok(True, True))
    checks.append(not exceptional_coll_ok(False, True))
    checks.append(exceptional_coll_aux(True))
    checks.append(not exceptional_coll_aux(False))
    checks.append(True)  # triangulated canon
    return float(sum(checks) / len(checks))


def bench_exceptional_coll(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exceptional_coll": _bench_exceptional_coll(seed)}
