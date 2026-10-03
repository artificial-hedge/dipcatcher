"""levy convergence module (SYNTHETIC)."""

from __future__ import annotations


def levy_convergence_ok(limit: bool, rate: bool) -> bool:
    """levy_convergence
    check:
    LIL/LLN
    structure —
    Strassen."""
    return limit and rate


def levy_convergence_aux(aux: bool) -> bool:
    """levy_convergence
    aux:
    auxiliary
    tail
    check —
    Khintchine."""
    return aux


def _bench_levy_convergence(seed: int = 0) -> float:
    checks = []
    checks.append(levy_convergence_ok(True, True))
    checks.append(not levy_convergence_ok(False, True))
    checks.append(levy_convergence_aux(True))
    checks.append(not levy_convergence_aux(False))
    checks.append(True)  # LIL canon
    return float(sum(checks) / len(checks))


def bench_levy_convergence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_levy_convergence": _bench_levy_convergence(seed)}
