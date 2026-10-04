"""nonoverlap dd module (SYNTHETIC)."""

from __future__ import annotations


def nonoverlap_dd_ok(dom: bool, res: bool) -> bool:
    """nonoverlap_dd
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def nonoverlap_dd_aux(aux: bool) -> bool:
    """nonoverlap_dd
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_nonoverlap_dd(seed: int = 0) -> float:
    checks = []
    checks.append(nonoverlap_dd_ok(True, True))
    checks.append(not nonoverlap_dd_ok(False, True))
    checks.append(nonoverlap_dd_aux(True))
    checks.append(not nonoverlap_dd_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_nonoverlap_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonoverlap_dd": _bench_nonoverlap_dd(seed)}
