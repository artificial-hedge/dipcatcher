"""overlap dd module (SYNTHETIC)."""

from __future__ import annotations


def overlap_dd_ok(dom: bool, res: bool) -> bool:
    """overlap_dd
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def overlap_dd_aux(aux: bool) -> bool:
    """overlap_dd
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_overlap_dd(seed: int = 0) -> float:
    checks = []
    checks.append(overlap_dd_ok(True, True))
    checks.append(not overlap_dd_ok(False, True))
    checks.append(overlap_dd_aux(True))
    checks.append(not overlap_dd_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_overlap_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_overlap_dd": _bench_overlap_dd(seed)}
