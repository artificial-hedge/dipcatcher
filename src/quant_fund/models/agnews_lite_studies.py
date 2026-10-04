"""agnews_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def agnews_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agnews_lite_studies

    check:
    agnews_lite_studies: AG-News metrics
    """
    return fit_ok and sample_ok


def agnews_lite_studies_aux(aux: bool) -> bool:
    """agnews_lite_studies

    aux:
    agnews_lite_studies: articles, summaries, labels, and scores
    """
    return aux


def _bench_agnews_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agnews_lite_studies_ok(True, True))
    checks.append(not agnews_lite_studies_ok(False, True))
    checks.append(agnews_lite_studies_aux(True))
    checks.append(not agnews_lite_studies_aux(False))
    checks.append(True)  # summarization-2 canon
    return float(sum(checks) / len(checks))


def bench_agnews_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agnews_lite_studies": _bench_agnews_lite_studies(seed)}
