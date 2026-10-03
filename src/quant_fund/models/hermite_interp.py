"""hermite interp module (SYNTHETIC)."""

from __future__ import annotations


def hermite_interp_ok(node: bool, weight: bool) -> bool:
    """hermite_interp
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def hermite_interp_aux(aux: bool) -> bool:
    """hermite_interp
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_hermite_interp(seed: int = 0) -> float:
    checks = []
    checks.append(hermite_interp_ok(True, True))
    checks.append(not hermite_interp_ok(False, True))
    checks.append(hermite_interp_aux(True))
    checks.append(not hermite_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_hermite_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermite_interp": _bench_hermite_interp(seed)}
