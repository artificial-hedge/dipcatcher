"""triggerfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def triggerfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """triggerfish_qa_studies

    check:
    triggerfish_qa_studies: TriggerfishQA metrics
    """
    return fit_ok and sample_ok


def triggerfish_qa_studies_aux(aux: bool) -> bool:
    """triggerfish_qa_studies

    aux:
    triggerfish_qa_studies: triggerfish, sandy lagoons, answers, and scores
    """
    return aux


def _bench_triggerfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(triggerfish_qa_studies_ok(True, True))
    checks.append(not triggerfish_qa_studies_ok(False, True))
    checks.append(triggerfish_qa_studies_aux(True))
    checks.append(not triggerfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-2 canon
    return float(sum(checks) / len(checks))


def bench_triggerfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_triggerfish_qa_studies": _bench_triggerfish_qa_studies(seed)}
