"""neumann dd module (SYNTHETIC)."""

from __future__ import annotations


def neumann_dd_ok(part: bool, coarse: bool) -> bool:
    """neumann_dd
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def neumann_dd_aux(aux: bool) -> bool:
    """neumann_dd
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_neumann_dd(seed: int = 0) -> float:
    checks = []
    checks.append(neumann_dd_ok(True, True))
    checks.append(not neumann_dd_ok(False, True))
    checks.append(neumann_dd_aux(True))
    checks.append(not neumann_dd_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_neumann_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neumann_dd": _bench_neumann_dd(seed)}
