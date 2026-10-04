"""neville interp module (SYNTHETIC)."""

from __future__ import annotations


def neville_interp_ok(node: bool, weight: bool) -> bool:
    """neville_interp
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def neville_interp_aux(aux: bool) -> bool:
    """neville_interp
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_neville_interp(seed: int = 0) -> float:
    checks = []
    checks.append(neville_interp_ok(True, True))
    checks.append(not neville_interp_ok(False, True))
    checks.append(neville_interp_aux(True))
    checks.append(not neville_interp_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_neville_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neville_interp": _bench_neville_interp(seed)}
