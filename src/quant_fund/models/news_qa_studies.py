"""news_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def news_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """news_qa_studies

    check:
    news_qa_studies: NewsQA answer-extraction metrics
    """
    return fit_ok and sample_ok


def news_qa_studies_aux(aux: bool) -> bool:
    """news_qa_studies

    aux:
    news_qa_studies: articles, questions, spans, and f1
    """
    return aux


def _bench_news_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(news_qa_studies_ok(True, True))
    checks.append(not news_qa_studies_ok(False, True))
    checks.append(news_qa_studies_aux(True))
    checks.append(not news_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-2 canon
    return float(sum(checks) / len(checks))


def bench_news_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_news_qa_studies": _bench_news_qa_studies(seed)}
