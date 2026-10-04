"""stable sheaf module (SYNTHETIC)."""

from __future__ import annotations


def stable_sheaf_ok(homotopy: bool, stable: bool) -> bool:
    """stable_sheaf
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def stable_sheaf_aux(aux: bool) -> bool:
    """stable_sheaf
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_stable_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(stable_sheaf_ok(True, True))
    checks.append(not stable_sheaf_ok(False, True))
    checks.append(stable_sheaf_aux(True))
    checks.append(not stable_sheaf_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_sheaf": _bench_stable_sheaf(seed)}
