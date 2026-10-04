"""asdiv_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def asdiv_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """asdiv_lite_studies

    check:
    asdiv_lite_studies: ASDiv arithmetic metrics
    """
    return fit_ok and sample_ok


def asdiv_lite_studies_aux(aux: bool) -> bool:
    """asdiv_lite_studies

    aux:
    asdiv_lite_studies: problems, expressions, answers, and scores
    """
    return aux


def _bench_asdiv_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(asdiv_lite_studies_ok(True, True))
    checks.append(not asdiv_lite_studies_ok(False, True))
    checks.append(asdiv_lite_studies_aux(True))
    checks.append(not asdiv_lite_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_asdiv_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asdiv_lite_studies": _bench_asdiv_lite_studies(seed)}
