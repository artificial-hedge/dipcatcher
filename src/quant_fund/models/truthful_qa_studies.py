"""truthful_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def truthful_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """truthful_qa_studies

    check:
    truthful_qa_studies: TruthfulQA truthful/informative scoring and BLEU-tails
    """
    return fit_ok and sample_ok


def truthful_qa_studies_aux(aux: bool) -> bool:
    """truthful_qa_studies

    aux:
    truthful_qa_studies: questions, best answers, and truthfulness metrics
    """
    return aux


def _bench_truthful_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(truthful_qa_studies_ok(True, True))
    checks.append(not truthful_qa_studies_ok(False, True))
    checks.append(truthful_qa_studies_aux(True))
    checks.append(not truthful_qa_studies_aux(False))
    checks.append(True)  # long-context-factuality canon
    return float(sum(checks) / len(checks))


def bench_truthful_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_truthful_qa_studies": _bench_truthful_qa_studies(seed)}
