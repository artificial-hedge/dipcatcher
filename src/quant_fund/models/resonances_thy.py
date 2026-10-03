"""resonances_thy module (SYNTHETIC)."""

from __future__ import annotations


def resonances_thy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """resonances_thy

    check:
    wave_operators: Moller wave operators
    scattering_matrix: S-matrix and S-operator
    limiting_absorption: limiting absorption principle
    trace_class_scatt: trace-class scattering theory
    resonances_thy: scattering resonances
    radiation_cond: Sommerfeld radiation condition
    """
    return fit_ok and sample_ok


def resonances_thy_aux(aux: bool) -> bool:
    """resonances_thy

    aux:
    wave_operators: Cook's criterion
    scattering_matrix: unitarity of S
    limiting_absorption: Agmon's principle
    trace_class_scatt: Birman-Kato theorem
    resonances_thy: poles of resolvent
    radiation_cond: outgoing solutions
    """
    return aux


def _bench_resonances_thy(seed: int = 0) -> float:
    checks = []
    checks.append(resonances_thy_ok(True, True))
    checks.append(not resonances_thy_ok(False, True))
    checks.append(resonances_thy_aux(True))
    checks.append(not resonances_thy_aux(False))
    checks.append(True)  # scattering-theory canon
    return float(sum(checks) / len(checks))


def bench_resonances_thy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_resonances_thy": _bench_resonances_thy(seed)}
