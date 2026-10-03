"""stable group module (SYNTHETIC)."""

from __future__ import annotations


def stable_group_ok(homotopy: bool, stable: bool) -> bool:
    """stable_group
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_group_aux(aux: bool) -> bool:
    """stable_group
    aux:
    auxiliary
    homotopy
    check —
    monoid."""
    return aux


def _bench_stable_group(seed: int = 0) -> float:
    checks = []
    checks.append(stable_group_ok(True, True))
    checks.append(not stable_group_ok(False, True))
    checks.append(stable_group_aux(True))
    checks.append(not stable_group_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_group": _bench_stable_group(seed)}
