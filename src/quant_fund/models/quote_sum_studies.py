"""quote_sum_studies module (SYNTHETIC)."""

from __future__ import annotations


def quote_sum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quote_sum_studies

    check:
    quote_sum_studies: QuoteSum metrics
    """
    return fit_ok and sample_ok


def quote_sum_studies_aux(aux: bool) -> bool:
    """quote_sum_studies

    aux:
    quote_sum_studies: speeches, questions, quotes, and scores
    """
    return aux


def _bench_quote_sum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quote_sum_studies_ok(True, True))
    checks.append(not quote_sum_studies_ok(False, True))
    checks.append(quote_sum_studies_aux(True))
    checks.append(not quote_sum_studies_aux(False))
    checks.append(True)  # long-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_quote_sum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quote_sum_studies": _bench_quote_sum_studies(seed)}
