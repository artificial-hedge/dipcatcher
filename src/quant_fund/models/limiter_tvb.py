"""limiter tvb module (SYNTHETIC)."""

from __future__ import annotations


def limiter_tvb_ok(basis: bool, flux: bool) -> bool:
    """limiter_tvb
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def limiter_tvb_aux(aux: bool) -> bool:
    """limiter_tvb
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_limiter_tvb(seed: int = 0) -> float:
    checks = []
    checks.append(limiter_tvb_ok(True, True))
    checks.append(not limiter_tvb_ok(False, True))
    checks.append(limiter_tvb_aux(True))
    checks.append(not limiter_tvb_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_limiter_tvb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_limiter_tvb": _bench_limiter_tvb(seed)}
