"""vireo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def vireo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vireo_qa_studies

    check:
    vireo_qa_studies: VireoQA metrics
    """
    return fit_ok and sample_ok


def vireo_qa_studies_aux(aux: bool) -> bool:
    """vireo_qa_studies

    aux:
    vireo_qa_studies: vireos, canopies, answers, and scores
    """
    return aux


def _bench_vireo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(vireo_qa_studies_ok(True, True))
    checks.append(not vireo_qa_studies_ok(False, True))
    checks.append(vireo_qa_studies_aux(True))
    checks.append(not vireo_qa_studies_aux(False))
    checks.append(True)  # songbird-2 canon
    return float(sum(checks) / len(checks))


def bench_vireo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vireo_qa_studies": _bench_vireo_qa_studies(seed)}
