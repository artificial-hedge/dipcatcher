"""darner_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def darner_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """darner_qa_studies

    check:
    darner_qa_studies: DarnerQA metrics
    """
    return fit_ok and sample_ok


def darner_qa_studies_aux(aux: bool) -> bool:
    """darner_qa_studies

    aux:
    darner_qa_studies: darners, ponds, answers, and scores
    """
    return aux


def _bench_darner_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(darner_qa_studies_ok(True, True))
    checks.append(not darner_qa_studies_ok(False, True))
    checks.append(darner_qa_studies_aux(True))
    checks.append(not darner_qa_studies_aux(False))
    checks.append(True)  # dragonfly canon
    return float(sum(checks) / len(checks))


def bench_darner_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_darner_qa_studies": _bench_darner_qa_studies(seed)}
