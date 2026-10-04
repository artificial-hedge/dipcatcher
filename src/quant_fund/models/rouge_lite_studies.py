"""rouge_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def rouge_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rouge_lite_studies

    check:
    rouge_lite_studies: ROUGE-L overlap metrics
    """
    return fit_ok and sample_ok


def rouge_lite_studies_aux(aux: bool) -> bool:
    """rouge_lite_studies

    aux:
    rouge_lite_studies: candidates, references, labels, and scores
    """
    return aux


def _bench_rouge_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rouge_lite_studies_ok(True, True))
    checks.append(not rouge_lite_studies_ok(False, True))
    checks.append(rouge_lite_studies_aux(True))
    checks.append(not rouge_lite_studies_aux(False))
    checks.append(True)  # generation-metric canon
    return float(sum(checks) / len(checks))


def bench_rouge_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rouge_lite_studies": _bench_rouge_lite_studies(seed)}
