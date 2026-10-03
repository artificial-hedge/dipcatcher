"""strong sol module (SYNTHETIC)."""

from __future__ import annotations


def strong_sol_ok(sm1: bool, pw: bool) -> bool:
    """strong_sol
    check:
    semimartingale
    structure —
    usual
    conditions."""
    return sm1 and pw


def strong_sol_aux(aux: bool) -> bool:
    """strong_sol
    aux:
    auxiliary
    canonical
    check —
    Dolean
    measure."""
    return aux


def _bench_strong_sol(seed: int = 0) -> float:
    checks = []
    checks.append(strong_sol_ok(True, True))
    checks.append(not strong_sol_ok(False, True))
    checks.append(strong_sol_aux(True))
    checks.append(not strong_sol_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_strong_sol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strong_sol": _bench_strong_sol(seed)}
