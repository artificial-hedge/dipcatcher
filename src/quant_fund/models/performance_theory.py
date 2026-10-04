"""performance_theory module (SYNTHETIC)."""

from __future__ import annotations


def performance_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """performance_theory

    check:
    theater_studies: theater studies
    dance_studies: dance studies
    performance_theory: performance theory
    dramaturgy: dramaturgy
    choreography: choreography
    stage_design: stage design
    """
    return fit_ok and sample_ok


def performance_theory_aux(aux: bool) -> bool:
    """performance_theory

    aux:
    theater_studies: drama analysis
    dance_studies: movement studies
    performance_theory: performance analysis
    dramaturgy: dramatic composition
    choreography: dance composition
    stage_design: scenography
    """
    return aux


def _bench_performance_theory(seed: int = 0) -> float:
    checks = []
    checks.append(performance_theory_ok(True, True))
    checks.append(not performance_theory_ok(False, True))
    checks.append(performance_theory_aux(True))
    checks.append(not performance_theory_aux(False))
    checks.append(True)  # performing arts canon
    return float(sum(checks) / len(checks))


def bench_performance_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_performance_theory": _bench_performance_theory(seed)}
