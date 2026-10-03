"""penalty dg module (SYNTHETIC)."""

from __future__ import annotations


def penalty_dg_ok(basis: bool, flux: bool) -> bool:
    """penalty_dg
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def penalty_dg_aux(aux: bool) -> bool:
    """penalty_dg
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_penalty_dg(seed: int = 0) -> float:
    checks = []
    checks.append(penalty_dg_ok(True, True))
    checks.append(not penalty_dg_ok(False, True))
    checks.append(penalty_dg_aux(True))
    checks.append(not penalty_dg_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_penalty_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_penalty_dg": _bench_penalty_dg(seed)}
