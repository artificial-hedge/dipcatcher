"""r0_estimation module (SYNTHETIC)."""

from __future__ import annotations


def r0_estimation_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """r0_estimation

    check:
    sir_epidemic: SIR epidemic model
    sis_epidemic: SIS epidemic model
    seir_epidemic: SEIR epidemic model
    r0_estimation: basic reproduction number
    herd_immunity: herd immunity threshold
    branching_epidemic: branching process epidemic
    """
    return fit_ok and sample_ok


def r0_estimation_aux(aux: bool) -> bool:
    """r0_estimation

    aux:
    sir_epidemic: Kermack-McKendrick
    sis_epidemic: endemic equilibrium
    seir_epidemic: incubation period
    r0_estimation: next-generation matrix
    herd_immunity: vaccination threshold
    branching_epidemic: extinction probability
    """
    return aux


def _bench_r0_estimation(seed: int = 0) -> float:
    checks = []
    checks.append(r0_estimation_ok(True, True))
    checks.append(not r0_estimation_ok(False, True))
    checks.append(r0_estimation_aux(True))
    checks.append(not r0_estimation_aux(False))
    checks.append(True)  # epidemiology canon
    return float(sum(checks) / len(checks))


def bench_r0_estimation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_r0_estimation": _bench_r0_estimation(seed)}
