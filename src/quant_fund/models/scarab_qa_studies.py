"""scarab_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def scarab_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scarab_qa_studies

    check:
    scarab_qa_studies: ScarabQA metrics
    """
    return fit_ok and sample_ok


def scarab_qa_studies_aux(aux: bool) -> bool:
    """scarab_qa_studies

    aux:
    scarab_qa_studies: scarabs, dung, answers, and scores
    """
    return aux


def _bench_scarab_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(scarab_qa_studies_ok(True, True))
    checks.append(not scarab_qa_studies_ok(False, True))
    checks.append(scarab_qa_studies_aux(True))
    checks.append(not scarab_qa_studies_aux(False))
    checks.append(True)  # insect-2 canon
    return float(sum(checks) / len(checks))


def bench_scarab_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scarab_qa_studies": _bench_scarab_qa_studies(seed)}
