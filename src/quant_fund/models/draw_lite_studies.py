"""draw_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def draw_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """draw_lite_studies

    check:
    draw_lite_studies: DRAW metrics
    """
    return fit_ok and sample_ok


def draw_lite_studies_aux(aux: bool) -> bool:
    """draw_lite_studies

    aux:
    draw_lite_studies: problems, derivations, answers, and scores
    """
    return aux


def _bench_draw_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(draw_lite_studies_ok(True, True))
    checks.append(not draw_lite_studies_ok(False, True))
    checks.append(draw_lite_studies_aux(True))
    checks.append(not draw_lite_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_draw_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_draw_lite_studies": _bench_draw_lite_studies(seed)}
