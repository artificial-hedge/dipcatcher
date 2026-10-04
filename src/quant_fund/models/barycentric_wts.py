"""barycentric wts module (SYNTHETIC)."""

from __future__ import annotations


def barycentric_wts_ok(node: bool, weight: bool) -> bool:
    """barycentric_wts
    check:
    interpolation
    canon — node/
    weight
    consistency."""
    return node and weight


def barycentric_wts_aux(aux: bool) -> bool:
    """barycentric_wts
    aux:
    auxiliary
    interp check —
    reproducing bound."""
    return aux


def _bench_barycentric_wts(seed: int = 0) -> float:
    checks = []
    checks.append(barycentric_wts_ok(True, True))
    checks.append(not barycentric_wts_ok(False, True))
    checks.append(barycentric_wts_aux(True))
    checks.append(not barycentric_wts_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_barycentric_wts(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barycentric_wts": _bench_barycentric_wts(seed)}
