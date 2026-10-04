"""multi_news_studies module (SYNTHETIC)."""

from __future__ import annotations


def multi_news_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """multi_news_studies

    check:
    multi_news_studies: Multi-News summarization metrics
    """
    return fit_ok and sample_ok


def multi_news_studies_aux(aux: bool) -> bool:
    """multi_news_studies

    aux:
    multi_news_studies: documents, summaries, references, and scores
    """
    return aux


def _bench_multi_news_studies(seed: int = 0) -> float:
    checks = []
    checks.append(multi_news_studies_ok(True, True))
    checks.append(not multi_news_studies_ok(False, True))
    checks.append(multi_news_studies_aux(True))
    checks.append(not multi_news_studies_aux(False))
    checks.append(True)  # summarization canon
    return float(sum(checks) / len(checks))


def bench_multi_news_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multi_news_studies": _bench_multi_news_studies(seed)}
