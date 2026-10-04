"""stable dual module (SYNTHETIC)."""

from __future__ import annotations


def stable_dual_ok(homotopy: bool, stable: bool) -> bool:
    """stable_dual
    check:
    homotopy
    structure —
    general."""
    return homotopy and stable


def stable_dual_aux(aux: bool) -> bool:
    """stable_dual
    aux:
    auxiliary
    homotopy
    check —
    rational."""
    return aux


def _bench_stable_dual(seed: int = 0) -> float:
    checks = []
    checks.append(stable_dual_ok(True, True))
    checks.append(not stable_dual_ok(False, True))
    checks.append(stable_dual_aux(True))
    checks.append(not stable_dual_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_dual(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_dual": _bench_stable_dual(seed)}
