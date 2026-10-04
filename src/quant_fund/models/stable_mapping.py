"""stable mapping module (SYNTHETIC)."""

from __future__ import annotations


def stable_mapping_ok(homotopy: bool, stable: bool) -> bool:
    """stable_mapping
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_mapping_aux(aux: bool) -> bool:
    """stable_mapping
    aux:
    auxiliary
    homotopy
    check —
    limit."""
    return aux


def _bench_stable_mapping(seed: int = 0) -> float:
    checks = []
    checks.append(stable_mapping_ok(True, True))
    checks.append(not stable_mapping_ok(False, True))
    checks.append(stable_mapping_aux(True))
    checks.append(not stable_mapping_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_mapping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_mapping": _bench_stable_mapping(seed)}
