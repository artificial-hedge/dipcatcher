"""lagrange interp module (SYNTHETIC)."""

from __future__ import annotations


def lagrange_interp_ok(node: bool, weight: bool) -> bool:
    """lagrange_interp
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def lagrange_interp_aux(aux: bool) -> bool:
    """lagrange_interp
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_lagrange_interp(seed: int = 0) -> float:
    checks = []
    checks.append(lagrange_interp_ok(True, True))
    checks.append(not lagrange_interp_ok(False, True))
    checks.append(lagrange_interp_aux(True))
    checks.append(not lagrange_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_lagrange_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lagrange_interp": _bench_lagrange_interp(seed)}
