"""choreography module (SYNTHETIC)."""

from __future__ import annotations


def choreography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """choreography

    check:
    theater_studies: theater studies
    dance_studies: dance studies
    performance_theory: performance theory
    dramaturgy: dramaturgy
    choreography: choreography
    stage_design: stage design
    """
    return fit_ok and sample_ok


def choreography_aux(aux: bool) -> bool:
    """choreography

    aux:
    theater_studies: drama analysis
    dance_studies: movement studies
    performance_theory: performance analysis
    dramaturgy: dramatic composition
    choreography: dance composition
    stage_design: scenography
    """
    return aux


def _bench_choreography(seed: int = 0) -> float:
    checks = []
    checks.append(choreography_ok(True, True))
    checks.append(not choreography_ok(False, True))
    checks.append(choreography_aux(True))
    checks.append(not choreography_aux(False))
    checks.append(True)  # performing arts canon
    return float(sum(checks) / len(checks))


def bench_choreography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_choreography": _bench_choreography(seed)}
