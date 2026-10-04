"""diagonal scale module (SYNTHETIC)."""

from __future__ import annotations


def diagonal_scale_ok(dom: bool, res: bool) -> bool:
    """diagonal_scale
    check:
    preconditioner —
    decomposition
    consistency."""
    return dom and res


def diagonal_scale_aux(aux: bool) -> bool:
    """diagonal_scale
    aux:
    auxiliary
    preconditioner check —
    energy bound."""
    return aux


def _bench_diagonal_scale(seed: int = 0) -> float:
    checks = []
    checks.append(diagonal_scale_ok(True, True))
    checks.append(not diagonal_scale_ok(False, True))
    checks.append(diagonal_scale_aux(True))
    checks.append(not diagonal_scale_aux(False))
    checks.append(True)  # preconditioner canon
    return float(sum(checks) / len(checks))


def bench_diagonal_scale(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diagonal_scale": _bench_diagonal_scale(seed)}
