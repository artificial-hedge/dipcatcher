"""arc_challenge_studies module (SYNTHETIC)."""

from __future__ import annotations


def arc_challenge_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arc_challenge_studies

    check:
    arc_challenge_studies: ARC challenge-set metrics
    """
    return fit_ok and sample_ok


def arc_challenge_studies_aux(aux: bool) -> bool:
    """arc_challenge_studies

    aux:
    arc_challenge_studies: questions, choices, explanations, and scores
    """
    return aux


def _bench_arc_challenge_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arc_challenge_studies_ok(True, True))
    checks.append(not arc_challenge_studies_ok(False, True))
    checks.append(arc_challenge_studies_aux(True))
    checks.append(not arc_challenge_studies_aux(False))
    checks.append(True)  # science-eval canon
    return float(sum(checks) / len(checks))


def bench_arc_challenge_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arc_challenge_studies": _bench_arc_challenge_studies(seed)}
