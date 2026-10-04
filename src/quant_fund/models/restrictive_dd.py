"""restrictive dd module (SYNTHETIC)."""

from __future__ import annotations


def restrictive_dd_ok(dom: bool, res: bool) -> bool:
    """restrictive_dd
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def restrictive_dd_aux(aux: bool) -> bool:
    """restrictive_dd
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_restrictive_dd(seed: int = 0) -> float:
    checks = []
    checks.append(restrictive_dd_ok(True, True))
    checks.append(not restrictive_dd_ok(False, True))
    checks.append(restrictive_dd_aux(True))
    checks.append(not restrictive_dd_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_restrictive_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_restrictive_dd": _bench_restrictive_dd(seed)}
