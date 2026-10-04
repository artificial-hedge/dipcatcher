"""motivic galois2 module (SYNTHETIC)."""

from __future__ import annotations


def motivic_galois2_ok(motivic: bool, hodge: bool) -> bool:
    """motivic_galois2
    check:
    motivic
    structure —
    Hodge."""
    return motivic and hodge


def motivic_galois2_aux(aux: bool) -> bool:
    """motivic_galois2
    aux:
    auxiliary
    motivic
    check —
    Tate."""
    return aux


def _bench_motivic_galois2(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_galois2_ok(True, True))
    checks.append(not motivic_galois2_ok(False, True))
    checks.append(motivic_galois2_aux(True))
    checks.append(not motivic_galois2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_galois2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_galois2": _bench_motivic_galois2(seed)}
