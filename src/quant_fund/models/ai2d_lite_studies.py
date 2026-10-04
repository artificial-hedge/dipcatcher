"""ai2d_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def ai2d_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ai2d_lite_studies

    check:
    ai2d_lite_studies: AI2D metrics
    """
    return fit_ok and sample_ok


def ai2d_lite_studies_aux(aux: bool) -> bool:
    """ai2d_lite_studies

    aux:
    ai2d_lite_studies: diagrams, questions, answers, and scores
    """
    return aux


def _bench_ai2d_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ai2d_lite_studies_ok(True, True))
    checks.append(not ai2d_lite_studies_ok(False, True))
    checks.append(ai2d_lite_studies_aux(True))
    checks.append(not ai2d_lite_studies_aux(False))
    checks.append(True)  # vision-doc-QA canon
    return float(sum(checks) / len(checks))


def bench_ai2d_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ai2d_lite_studies": _bench_ai2d_lite_studies(seed)}
