"""rouge_we_studies module (SYNTHETIC)."""

from __future__ import annotations


def rouge_we_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rouge_we_studies

    check:
    rouge_we_studies: ROUGE-WE embedding metrics
    """
    return fit_ok and sample_ok


def rouge_we_studies_aux(aux: bool) -> bool:
    """rouge_we_studies

    aux:
    rouge_we_studies: references, candidates, embeddings, and scores
    """
    return aux


def _bench_rouge_we_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rouge_we_studies_ok(True, True))
    checks.append(not rouge_we_studies_ok(False, True))
    checks.append(rouge_we_studies_aux(True))
    checks.append(not rouge_we_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_rouge_we_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rouge_we_studies": _bench_rouge_we_studies(seed)}
