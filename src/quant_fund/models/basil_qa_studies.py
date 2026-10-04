"""basil_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def basil_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """basil_qa_studies

    check:
    basil_qa_studies: BasilQA metrics
    """
    return fit_ok and sample_ok


def basil_qa_studies_aux(aux: bool) -> bool:
    """basil_qa_studies

    aux:
    basil_qa_studies: basils, leaves, answers, and scores
    """
    return aux


def _bench_basil_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(basil_qa_studies_ok(True, True))
    checks.append(not basil_qa_studies_ok(False, True))
    checks.append(basil_qa_studies_aux(True))
    checks.append(not basil_qa_studies_aux(False))
    checks.append(True)  # herb canon
    return float(sum(checks) / len(checks))


def bench_basil_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_basil_qa_studies": _bench_basil_qa_studies(seed)}
