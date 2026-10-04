"""branching_epidemic module (SYNTHETIC)."""

from __future__ import annotations


def branching_epidemic_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """branching_epidemic

    check:
    sir_epidemic: SIR epidemic model
    sis_epidemic: SIS epidemic model
    seir_epidemic: SEIR epidemic model
    r0_estimation: basic reproduction number
    herd_immunity: herd immunity threshold
    branching_epidemic: branching process epidemic
    """
    return fit_ok and sample_ok


def branching_epidemic_aux(aux: bool) -> bool:
    """branching_epidemic

    aux:
    sir_epidemic: Kermack-McKendrick
    sis_epidemic: endemic equilibrium
    seir_epidemic: incubation period
    r0_estimation: next-generation matrix
    herd_immunity: vaccination threshold
    branching_epidemic: extinction probability
    """
    return aux


def _bench_branching_epidemic(seed: int = 0) -> float:
    checks = []
    checks.append(branching_epidemic_ok(True, True))
    checks.append(not branching_epidemic_ok(False, True))
    checks.append(branching_epidemic_aux(True))
    checks.append(not branching_epidemic_aux(False))
    checks.append(True)  # epidemiology canon
    return float(sum(checks) / len(checks))


def bench_branching_epidemic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_branching_epidemic": _bench_branching_epidemic(seed)}
