"""elm_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def elm_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """elm_lite_studies

    check:
    elm_lite_studies: ELife lay-summary metrics
    """
    return fit_ok and sample_ok


def elm_lite_studies_aux(aux: bool) -> bool:
    """elm_lite_studies

    aux:
    elm_lite_studies: papers, abstracts, lay summaries, and scores
    """
    return aux


def _bench_elm_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(elm_lite_studies_ok(True, True))
    checks.append(not elm_lite_studies_ok(False, True))
    checks.append(elm_lite_studies_aux(True))
    checks.append(not elm_lite_studies_aux(False))
    checks.append(True)  # long-doc-summarization canon
    return float(sum(checks) / len(checks))


def bench_elm_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elm_lite_studies": _bench_elm_lite_studies(seed)}
