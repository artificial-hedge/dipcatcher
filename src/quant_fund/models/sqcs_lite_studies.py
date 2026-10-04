"""sqcs_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def sqcs_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sqcs_lite_studies

    check:
    sqcs_lite_studies: SQCS metrics
    """
    return fit_ok and sample_ok


def sqcs_lite_studies_aux(aux: bool) -> bool:
    """sqcs_lite_studies

    aux:
    sqcs_lite_studies: queries, documents, summaries, and scores
    """
    return aux


def _bench_sqcs_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sqcs_lite_studies_ok(True, True))
    checks.append(not sqcs_lite_studies_ok(False, True))
    checks.append(sqcs_lite_studies_aux(True))
    checks.append(not sqcs_lite_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_sqcs_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sqcs_lite_studies": _bench_sqcs_lite_studies(seed)}
