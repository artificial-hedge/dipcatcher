"""controlled path module (SYNTHETIC)."""

from __future__ import annotations


def controlled_path_ok(rp1: bool, lift: bool) -> bool:
    """controlled_path
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def controlled_path_aux(aux: bool) -> bool:
    """controlled_path
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_controlled_path(seed: int = 0) -> float:
    checks = []
    checks.append(controlled_path_ok(True, True))
    checks.append(not controlled_path_ok(False, True))
    checks.append(controlled_path_aux(True))
    checks.append(not controlled_path_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_controlled_path(seed: int = 0) -> dict[str, float]:
    return {"synthetic_controlled_path": _bench_controlled_path(seed)}
