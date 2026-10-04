"""dolphin_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def dolphin_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dolphin_lite_studies

    check:
    dolphin_lite_studies: Dolphin1878 metrics
    """
    return fit_ok and sample_ok


def dolphin_lite_studies_aux(aux: bool) -> bool:
    """dolphin_lite_studies

    aux:
    dolphin_lite_studies: problems, equations, answers, and scores
    """
    return aux


def _bench_dolphin_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dolphin_lite_studies_ok(True, True))
    checks.append(not dolphin_lite_studies_ok(False, True))
    checks.append(dolphin_lite_studies_aux(True))
    checks.append(not dolphin_lite_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_dolphin_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dolphin_lite_studies": _bench_dolphin_lite_studies(seed)}
