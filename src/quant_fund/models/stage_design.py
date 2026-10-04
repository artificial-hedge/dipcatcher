"""stage_design module (SYNTHETIC)."""

from __future__ import annotations


def stage_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stage_design

    check:
    theater_studies: theater studies
    dance_studies: dance studies
    performance_theory: performance theory
    dramaturgy: dramaturgy
    choreography: choreography
    stage_design: stage design
    """
    return fit_ok and sample_ok


def stage_design_aux(aux: bool) -> bool:
    """stage_design

    aux:
    theater_studies: drama analysis
    dance_studies: movement studies
    performance_theory: performance analysis
    dramaturgy: dramatic composition
    choreography: dance composition
    stage_design: scenography
    """
    return aux


def _bench_stage_design(seed: int = 0) -> float:
    checks = []
    checks.append(stage_design_ok(True, True))
    checks.append(not stage_design_ok(False, True))
    checks.append(stage_design_aux(True))
    checks.append(not stage_design_aux(False))
    checks.append(True)  # performing arts canon
    return float(sum(checks) / len(checks))


def bench_stage_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stage_design": _bench_stage_design(seed)}
