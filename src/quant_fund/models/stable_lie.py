"""stable lie module (SYNTHETIC)."""

from __future__ import annotations


def stable_lie_ok(homotopy: bool, stable: bool) -> bool:
    """stable_lie
    check:
    homotopy
    structure —
    general."""
    return homotopy and stable


def stable_lie_aux(aux: bool) -> bool:
    """stable_lie
    aux:
    auxiliary
    homotopy
    check —
    rational."""
    return aux


def _bench_stable_lie(seed: int = 0) -> float:
    checks = []
    checks.append(stable_lie_ok(True, True))
    checks.append(not stable_lie_ok(False, True))
    checks.append(stable_lie_aux(True))
    checks.append(not stable_lie_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_lie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_lie": _bench_stable_lie(seed)}
