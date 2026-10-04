"""amber_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def amber_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """amber_qa_studies

    check:
    amber_qa_studies: AmberQA metrics
    """
    return fit_ok and sample_ok


def amber_qa_studies_aux(aux: bool) -> bool:
    """amber_qa_studies

    aux:
    amber_qa_studies: ambers, fossils, answers, and scores
    """
    return aux


def _bench_amber_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(amber_qa_studies_ok(True, True))
    checks.append(not amber_qa_studies_ok(False, True))
    checks.append(amber_qa_studies_aux(True))
    checks.append(not amber_qa_studies_aux(False))
    checks.append(True)  # gem canon
    return float(sum(checks) / len(checks))


def bench_amber_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amber_qa_studies": _bench_amber_qa_studies(seed)}
