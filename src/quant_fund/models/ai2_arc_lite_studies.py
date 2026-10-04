"""ai2_arc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def ai2_arc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ai2_arc_lite_studies

    check:
    ai2_arc_lite_studies: AI2-ARC metrics
    """
    return fit_ok and sample_ok


def ai2_arc_lite_studies_aux(aux: bool) -> bool:
    """ai2_arc_lite_studies

    aux:
    ai2_arc_lite_studies: questions, choices, answers, and scores
    """
    return aux


def _bench_ai2_arc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ai2_arc_lite_studies_ok(True, True))
    checks.append(not ai2_arc_lite_studies_ok(False, True))
    checks.append(ai2_arc_lite_studies_aux(True))
    checks.append(not ai2_arc_lite_studies_aux(False))
    checks.append(True)  # science-QA-2 canon
    return float(sum(checks) / len(checks))


def bench_ai2_arc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ai2_arc_lite_studies": _bench_ai2_arc_lite_studies(seed)}
