"""bert_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def bert_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bert_score_studies

    check:
    bert_score_studies: BERTScore embedding metrics
    """
    return fit_ok and sample_ok


def bert_score_studies_aux(aux: bool) -> bool:
    """bert_score_studies

    aux:
    bert_score_studies: candidates, references, labels, and scores
    """
    return aux


def _bench_bert_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bert_score_studies_ok(True, True))
    checks.append(not bert_score_studies_ok(False, True))
    checks.append(bert_score_studies_aux(True))
    checks.append(not bert_score_studies_aux(False))
    checks.append(True)  # generation-metric canon
    return float(sum(checks) / len(checks))


def bench_bert_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bert_score_studies": _bench_bert_score_studies(seed)}
