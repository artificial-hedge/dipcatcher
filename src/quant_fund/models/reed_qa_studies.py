"""reed_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def reed_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reed_qa_studies

    check:
    reed_qa_studies: ReedQA metrics
    """
    return fit_ok and sample_ok


def reed_qa_studies_aux(aux: bool) -> bool:
    """reed_qa_studies

    aux:
    reed_qa_studies: reeds, shorelines, answers, and scores
    """
    return aux


def _bench_reed_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reed_qa_studies_ok(True, True))
    checks.append(not reed_qa_studies_ok(False, True))
    checks.append(reed_qa_studies_aux(True))
    checks.append(not reed_qa_studies_aux(False))
    checks.append(True)  # sedge canon
    return float(sum(checks) / len(checks))


def bench_reed_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reed_qa_studies": _bench_reed_qa_studies(seed)}
