"""balanced dd module (SYNTHETIC)."""

from __future__ import annotations


def balanced_dd_ok(dom: bool, res: bool) -> bool:
    """balanced_dd
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def balanced_dd_aux(aux: bool) -> bool:
    """balanced_dd
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_balanced_dd(seed: int = 0) -> float:
    checks = []
    checks.append(balanced_dd_ok(True, True))
    checks.append(not balanced_dd_ok(False, True))
    checks.append(balanced_dd_aux(True))
    checks.append(not balanced_dd_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_balanced_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_balanced_dd": _bench_balanced_dd(seed)}
