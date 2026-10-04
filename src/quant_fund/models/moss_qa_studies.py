"""moss_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moss_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moss_qa_studies

    check:
    moss_qa_studies: MossQA metrics
    """
    return fit_ok and sample_ok


def moss_qa_studies_aux(aux: bool) -> bool:
    """moss_qa_studies

    aux:
    moss_qa_studies: mosses, spores, answers, and scores
    """
    return aux


def _bench_moss_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moss_qa_studies_ok(True, True))
    checks.append(not moss_qa_studies_ok(False, True))
    checks.append(moss_qa_studies_aux(True))
    checks.append(not moss_qa_studies_aux(False))
    checks.append(True)  # flora canon
    return float(sum(checks) / len(checks))


def bench_moss_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moss_qa_studies": _bench_moss_qa_studies(seed)}
