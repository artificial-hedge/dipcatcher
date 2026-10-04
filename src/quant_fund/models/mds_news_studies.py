"""mds_news_studies module (SYNTHETIC)."""

from __future__ import annotations


def mds_news_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mds_news_studies

    check:
    mds_news_studies: MultiNews metrics
    """
    return fit_ok and sample_ok


def mds_news_studies_aux(aux: bool) -> bool:
    """mds_news_studies

    aux:
    mds_news_studies: clusters, sources, summaries, and scores
    """
    return aux


def _bench_mds_news_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mds_news_studies_ok(True, True))
    checks.append(not mds_news_studies_ok(False, True))
    checks.append(mds_news_studies_aux(True))
    checks.append(not mds_news_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_mds_news_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mds_news_studies": _bench_mds_news_studies(seed)}
