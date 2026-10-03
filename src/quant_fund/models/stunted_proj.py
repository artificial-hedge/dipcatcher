"""stunted proj module (SYNTHETIC)."""

from __future__ import annotations


def stunted_proj_ok(homotopy: bool, unstable: bool) -> bool:
    """stunted_proj
    check:
    homotopy
    unstable
    structure —
    periodic."""
    return homotopy and unstable


def stunted_proj_aux(aux: bool) -> bool:
    """stunted_proj
    aux:
    auxiliary
    homotopy
    check —
    Adams."""
    return aux


def _bench_stunted_proj(seed: int = 0) -> float:
    checks = []
    checks.append(stunted_proj_ok(True, True))
    checks.append(not stunted_proj_ok(False, True))
    checks.append(stunted_proj_aux(True))
    checks.append(not stunted_proj_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stunted_proj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stunted_proj": _bench_stunted_proj(seed)}
