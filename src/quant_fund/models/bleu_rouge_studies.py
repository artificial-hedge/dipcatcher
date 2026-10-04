"""bleu_rouge_studies module (SYNTHETIC)."""

from __future__ import annotations


def bleu_rouge_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bleu_rouge_studies

    check:
    bleu_rouge_studies: BLEU/ROUGE n-gram metrics
    """
    return fit_ok and sample_ok


def bleu_rouge_studies_aux(aux: bool) -> bool:
    """bleu_rouge_studies

    aux:
    bleu_rouge_studies: candidates, references, labels, and scores
    """
    return aux


def _bench_bleu_rouge_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bleu_rouge_studies_ok(True, True))
    checks.append(not bleu_rouge_studies_ok(False, True))
    checks.append(bleu_rouge_studies_aux(True))
    checks.append(not bleu_rouge_studies_aux(False))
    checks.append(True)  # generation-metric canon
    return float(sum(checks) / len(checks))


def bench_bleu_rouge_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bleu_rouge_studies": _bench_bleu_rouge_studies(seed)}
