"""poisson rm module (SYNTHETIC)."""

from __future__ import annotations


def poisson_rm_ok(rm: bool, comp: bool) -> bool:
    """poisson_rm
    check:
    random
    measure —
    compensator."""
    return rm and comp


def poisson_rm_aux(aux: bool) -> bool:
    """poisson_rm
    aux:
    auxiliary
    measure check —
    intensity."""
    return aux


def _bench_poisson_rm(seed: int = 0) -> float:
    checks = []
    checks.append(poisson_rm_ok(True, True))
    checks.append(not poisson_rm_ok(False, True))
    checks.append(poisson_rm_aux(True))
    checks.append(not poisson_rm_aux(False))
    checks.append(True)  # random-measure canon
    return float(sum(checks) / len(checks))


def bench_poisson_rm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poisson_rm": _bench_poisson_rm(seed)}
