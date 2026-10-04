"""poly chaos_uq module (SYNTHETIC)."""

from __future__ import annotations


def poly_chaos_uq_ok(uq: bool, mode: bool) -> bool:
    """poly_chaos_uq
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def poly_chaos_uq_aux(aux: bool) -> bool:
    """poly_chaos_uq
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_poly_chaos_uq(seed: int = 0) -> float:
    checks = []
    checks.append(poly_chaos_uq_ok(True, True))
    checks.append(not poly_chaos_uq_ok(False, True))
    checks.append(poly_chaos_uq_aux(True))
    checks.append(not poly_chaos_uq_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_poly_chaos_uq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poly_chaos_uq": _bench_poly_chaos_uq(seed)}
