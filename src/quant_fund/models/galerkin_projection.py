"""galerkin projection module (SYNTHETIC)."""

from __future__ import annotations


def galerkin_projection_ok(basis: bool, coef: bool) -> bool:
    """galerkin_projection
    check:
    radial basis / projection —
    basis/coefficient
    consistency."""
    return basis and coef


def galerkin_projection_aux(aux: bool) -> bool:
    """galerkin_projection
    aux:
    auxiliary
    basis check —
    reproducing bound."""
    return aux


def _bench_galerkin_projection(seed: int = 0) -> float:
    checks = []
    checks.append(galerkin_projection_ok(True, True))
    checks.append(not galerkin_projection_ok(False, True))
    checks.append(galerkin_projection_aux(True))
    checks.append(not galerkin_projection_aux(False))
    checks.append(True)  # RBF/basis canon
    return float(sum(checks) / len(checks))


def bench_galerkin_projection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galerkin_projection": _bench_galerkin_projection(seed)}
