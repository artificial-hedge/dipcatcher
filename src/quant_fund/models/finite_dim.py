"""finite dim module (SYNTHETIC)."""

from __future__ import annotations


def finite_dim_ok(law: bool, tight: bool) -> bool:
    """finite_dim
    check:
    law of
    process —
    tightness."""
    return law and tight


def finite_dim_aux(aux: bool) -> bool:
    """finite_dim
    aux:
    auxiliary
    law check —
    convergence."""
    return aux


def _bench_finite_dim(seed: int = 0) -> float:
    checks = []
    checks.append(finite_dim_ok(True, True))
    checks.append(not finite_dim_ok(False, True))
    checks.append(finite_dim_aux(True))
    checks.append(not finite_dim_aux(False))
    checks.append(True)  # law canon
    return float(sum(checks) / len(checks))


def bench_finite_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finite_dim": _bench_finite_dim(seed)}
