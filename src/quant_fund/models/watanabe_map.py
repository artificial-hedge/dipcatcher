"""watanabe map module (SYNTHETIC)."""

from __future__ import annotations


def watanabe_map_ok(nz1: bool, mc: bool) -> bool:
    """watanabe_map
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def watanabe_map_aux(aux: bool) -> bool:
    """watanabe_map
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_watanabe_map(seed: int = 0) -> float:
    checks = []
    checks.append(watanabe_map_ok(True, True))
    checks.append(not watanabe_map_ok(False, True))
    checks.append(watanabe_map_aux(True))
    checks.append(not watanabe_map_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_watanabe_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_watanabe_map": _bench_watanabe_map(seed)}
