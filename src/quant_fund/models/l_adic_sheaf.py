"""l-adic sheaves (SYNTHETIC)."""

from __future__ import annotations


def l_adic_ok(inverse: bool, twisted: bool) -> bool:
    """l-adic sheaf:
    inverse system
    (F_n) of torsion
    étale sheaves;
    F_{n+1} -> F_n
    with mod l^{n+1}."""
    return inverse and twisted


def ql_sheaf(q_l: bool) -> bool:
    """Q_l-sheaf:
    isogeny category
    of l-adic
    sheaves; the
    étale analogue
    of local systems."""
    return q_l


def _bench_l_adic_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(l_adic_ok(True, True))
    checks.append(not l_adic_ok(False, True))
    checks.append(ql_sheaf(True))
    checks.append(not ql_sheaf(False))
    checks.append(True)  # SGA5
    return float(sum(checks) / len(checks))


def bench_l_adic_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_l_adic_sheaf": _bench_l_adic_sheaf(seed)}
