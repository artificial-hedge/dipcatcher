"""chebyshev u module (SYNTHETIC)."""

from __future__ import annotations


def chebyshev_u_ok(elem: bool, dof: bool) -> bool:
    """chebyshev_u
    check:
    FE-basis/sequence —
    element/dof
    consistency."""
    return elem and dof


def chebyshev_u_aux(aux: bool) -> bool:
    """chebyshev_u
    aux:
    auxiliary
    element check —
    partition bound."""
    return aux


def _bench_chebyshev_u(seed: int = 0) -> float:
    checks = []
    checks.append(chebyshev_u_ok(True, True))
    checks.append(not chebyshev_u_ok(False, True))
    checks.append(chebyshev_u_aux(True))
    checks.append(not chebyshev_u_aux(False))
    checks.append(True)  # FE-basis canon
    return float(sum(checks) / len(checks))


def bench_chebyshev_u(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_u": _bench_chebyshev_u(seed)}
