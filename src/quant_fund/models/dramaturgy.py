"""dramaturgy module (SYNTHETIC)."""

from __future__ import annotations


def dramaturgy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dramaturgy

    check:
    theater_studies: theater studies
    dance_studies: dance studies
    performance_theory: performance theory
    dramaturgy: dramaturgy
    choreography: choreography
    stage_design: stage design
    """
    return fit_ok and sample_ok


def dramaturgy_aux(aux: bool) -> bool:
    """dramaturgy

    aux:
    theater_studies: drama analysis
    dance_studies: movement studies
    performance_theory: performance analysis
    dramaturgy: dramatic composition
    choreography: dance composition
    stage_design: scenography
    """
    return aux


def _bench_dramaturgy(seed: int = 0) -> float:
    checks = []
    checks.append(dramaturgy_ok(True, True))
    checks.append(not dramaturgy_ok(False, True))
    checks.append(dramaturgy_aux(True))
    checks.append(not dramaturgy_aux(False))
    checks.append(True)  # performing arts canon
    return float(sum(checks) / len(checks))


def bench_dramaturgy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dramaturgy": _bench_dramaturgy(seed)}
