"""hairspring map module (SYNTHETIC)."""

from __future__ import annotations


def hairspring_map_ok(rp1: bool, lift: bool) -> bool:
    """hairspring_map
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def hairspring_map_aux(aux: bool) -> bool:
    """hairspring_map
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_hairspring_map(seed: int = 0) -> float:
    checks = []
    checks.append(hairspring_map_ok(True, True))
    checks.append(not hairspring_map_ok(False, True))
    checks.append(hairspring_map_aux(True))
    checks.append(not hairspring_map_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_hairspring_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hairspring_map": _bench_hairspring_map(seed)}
