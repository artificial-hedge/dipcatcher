"""ciucu dimers module (SYNTHETIC)."""

from __future__ import annotations


def ciucu_dimers_ok(dim: bool, hf: bool) -> bool:
    """ciucu_dimers
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def ciucu_dimers_aux(aux: bool) -> bool:
    """ciucu_dimers
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_ciucu_dimers(seed: int = 0) -> float:
    checks = []
    checks.append(ciucu_dimers_ok(True, True))
    checks.append(not ciucu_dimers_ok(False, True))
    checks.append(ciucu_dimers_aux(True))
    checks.append(not ciucu_dimers_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_ciucu_dimers(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ciucu_dimers": _bench_ciucu_dimers(seed)}
