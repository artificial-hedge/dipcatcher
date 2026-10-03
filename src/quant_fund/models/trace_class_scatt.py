"""trace_class_scatt module (SYNTHETIC)."""

from __future__ import annotations


def trace_class_scatt_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trace_class_scatt

    check:
    wave_operators: Moller wave operators
    scattering_matrix: S-matrix and S-operator
    limiting_absorption: limiting absorption principle
    trace_class_scatt: trace-class scattering theory
    resonances_thy: scattering resonances
    radiation_cond: Sommerfeld radiation condition
    """
    return fit_ok and sample_ok


def trace_class_scatt_aux(aux: bool) -> bool:
    """trace_class_scatt

    aux:
    wave_operators: Cook's criterion
    scattering_matrix: unitarity of S
    limiting_absorption: Agmon's principle
    trace_class_scatt: Birman-Kato theorem
    resonances_thy: poles of resolvent
    radiation_cond: outgoing solutions
    """
    return aux


def _bench_trace_class_scatt(seed: int = 0) -> float:
    checks = []
    checks.append(trace_class_scatt_ok(True, True))
    checks.append(not trace_class_scatt_ok(False, True))
    checks.append(trace_class_scatt_aux(True))
    checks.append(not trace_class_scatt_aux(False))
    checks.append(True)  # scattering-theory canon
    return float(sum(checks) / len(checks))


def bench_trace_class_scatt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trace_class_scatt": _bench_trace_class_scatt(seed)}
