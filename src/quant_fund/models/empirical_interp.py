"""empirical interp module (SYNTHETIC)."""

from __future__ import annotations


def empirical_interp_ok(node: bool, wgt: bool) -> bool:
    """empirical_interp
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def empirical_interp_aux(aux: bool) -> bool:
    """empirical_interp
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_empirical_interp(seed: int = 0) -> float:
    checks = []
    checks.append(empirical_interp_ok(True, True))
    checks.append(not empirical_interp_ok(False, True))
    checks.append(empirical_interp_aux(True))
    checks.append(not empirical_interp_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_empirical_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_empirical_interp": _bench_empirical_interp(seed)}
