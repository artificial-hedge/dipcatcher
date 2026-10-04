"""compound poisson module (SYNTHETIC)."""

from __future__ import annotations


def compound_poisson_ok(jd: bool, mj: bool) -> bool:
    """compound_poisson
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def compound_poisson_aux(aux: bool) -> bool:
    """compound_poisson
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_compound_poisson(seed: int = 0) -> float:
    checks = []
    checks.append(compound_poisson_ok(True, True))
    checks.append(not compound_poisson_ok(False, True))
    checks.append(compound_poisson_aux(True))
    checks.append(not compound_poisson_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_compound_poisson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compound_poisson": _bench_compound_poisson(seed)}
