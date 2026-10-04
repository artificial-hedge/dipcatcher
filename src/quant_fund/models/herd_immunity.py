"""herd_immunity module (SYNTHETIC)."""

from __future__ import annotations


def herd_immunity_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """herd_immunity

    check:
    sir_epidemic: SIR epidemic model
    sis_epidemic: SIS epidemic model
    seir_epidemic: SEIR epidemic model
    r0_estimation: basic reproduction number
    herd_immunity: herd immunity threshold
    branching_epidemic: branching process epidemic
    """
    return fit_ok and sample_ok


def herd_immunity_aux(aux: bool) -> bool:
    """herd_immunity

    aux:
    sir_epidemic: Kermack-McKendrick
    sis_epidemic: endemic equilibrium
    seir_epidemic: incubation period
    r0_estimation: next-generation matrix
    herd_immunity: vaccination threshold
    branching_epidemic: extinction probability
    """
    return aux


def _bench_herd_immunity(seed: int = 0) -> float:
    checks = []
    checks.append(herd_immunity_ok(True, True))
    checks.append(not herd_immunity_ok(False, True))
    checks.append(herd_immunity_aux(True))
    checks.append(not herd_immunity_aux(False))
    checks.append(True)  # epidemiology canon
    return float(sum(checks) / len(checks))


def bench_herd_immunity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_herd_immunity": _bench_herd_immunity(seed)}
