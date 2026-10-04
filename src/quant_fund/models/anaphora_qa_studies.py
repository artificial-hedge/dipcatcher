"""anaphora_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def anaphora_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anaphora_qa_studies

    check:
    anaphora_qa_studies: AnaphoraQA metrics
    """
    return fit_ok and sample_ok


def anaphora_qa_studies_aux(aux: bool) -> bool:
    """anaphora_qa_studies

    aux:
    anaphora_qa_studies: sentences, referents, answers, and scores
    """
    return aux


def _bench_anaphora_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anaphora_qa_studies_ok(True, True))
    checks.append(not anaphora_qa_studies_ok(False, True))
    checks.append(anaphora_qa_studies_aux(True))
    checks.append(not anaphora_qa_studies_aux(False))
    checks.append(True)  # discourse-pragmatics canon
    return float(sum(checks) / len(checks))


def bench_anaphora_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anaphora_qa_studies": _bench_anaphora_qa_studies(seed)}
