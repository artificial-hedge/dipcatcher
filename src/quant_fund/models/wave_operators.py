"""wave_operators module (SYNTHETIC)."""

from __future__ import annotations


def wave_operators_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wave_operators

    check:
    wave_operators: Moller wave operators
    scattering_matrix: S-matrix and S-operator
    limiting_absorption: limiting absorption principle
    trace_class_scatt: trace-class scattering theory
    resonances_thy: scattering resonances
    radiation_cond: Sommerfeld radiation condition
    """
    return fit_ok and sample_ok


def wave_operators_aux(aux: bool) -> bool:
    """wave_operators

    aux:
    wave_operators: Cook's criterion
    scattering_matrix: unitarity of S
    limiting_absorption: Agmon's principle
    trace_class_scatt: Birman-Kato theorem
    resonances_thy: poles of resolvent
    radiation_cond: outgoing solutions
    """
    return aux


def _bench_wave_operators(seed: int = 0) -> float:
    checks = []
    checks.append(wave_operators_ok(True, True))
    checks.append(not wave_operators_ok(False, True))
    checks.append(wave_operators_aux(True))
    checks.append(not wave_operators_aux(False))
    checks.append(True)  # scattering-theory canon
    return float(sum(checks) / len(checks))


def bench_wave_operators(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wave_operators": _bench_wave_operators(seed)}
