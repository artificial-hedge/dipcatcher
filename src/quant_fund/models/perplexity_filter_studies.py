"""perplexity_filter_studies module (SYNTHETIC)."""

from __future__ import annotations


def perplexity_filter_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """perplexity_filter_studies

    check:
    perplexity_filter_studies: ngram-model perplexity scoring/docs and cutoffs
    """
    return fit_ok and sample_ok


def perplexity_filter_studies_aux(aux: bool) -> bool:
    """perplexity_filter_studies

    aux:
    perplexity_filter_studies: KenLM-style quality ppl gates/scores and thresholds
    """
    return aux


def _bench_perplexity_filter_studies(seed: int = 0) -> float:
    checks = []
    checks.append(perplexity_filter_studies_ok(True, True))
    checks.append(not perplexity_filter_studies_ok(False, True))
    checks.append(perplexity_filter_studies_aux(True))
    checks.append(not perplexity_filter_studies_aux(False))
    checks.append(True)  # data-filtering/dedup canon
    return float(sum(checks) / len(checks))


def bench_perplexity_filter_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perplexity_filter_studies": _bench_perplexity_filter_studies(seed)}
