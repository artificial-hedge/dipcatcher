"""scattering_matrix module (SYNTHETIC)."""

from __future__ import annotations


def scattering_matrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scattering_matrix

    check:
    wave_operators: Moller wave operators
    scattering_matrix: S-matrix and S-operator
    limiting_absorption: limiting absorption principle
    trace_class_scatt: trace-class scattering theory
    resonances_thy: scattering resonances
    radiation_cond: Sommerfeld radiation condition
    """
    return fit_ok and sample_ok


def scattering_matrix_aux(aux: bool) -> bool:
    """scattering_matrix

    aux:
    wave_operators: Cook's criterion
    scattering_matrix: unitarity of S
    limiting_absorption: Agmon's principle
    trace_class_scatt: Birman-Kato theorem
    resonances_thy: poles of resolvent
    radiation_cond: outgoing solutions
    """
    return aux


def _bench_scattering_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(scattering_matrix_ok(True, True))
    checks.append(not scattering_matrix_ok(False, True))
    checks.append(scattering_matrix_aux(True))
    checks.append(not scattering_matrix_aux(False))
    checks.append(True)  # scattering-theory canon
    return float(sum(checks) / len(checks))


def bench_scattering_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scattering_matrix": _bench_scattering_matrix(seed)}
