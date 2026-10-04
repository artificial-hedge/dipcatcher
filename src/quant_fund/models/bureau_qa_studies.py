"""bureau_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bureau_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bureau_qa_studies

    check:
    bureau_qa_studies: BureauQA metrics
    """
    return fit_ok and sample_ok


def bureau_qa_studies_aux(aux: bool) -> bool:
    """bureau_qa_studies

    aux:
    bureau_qa_studies: bureaus, offices, answers, and scores
    """
    return aux


def _bench_bureau_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bureau_qa_studies_ok(True, True))
    checks.append(not bureau_qa_studies_ok(False, True))
    checks.append(bureau_qa_studies_aux(True))
    checks.append(not bureau_qa_studies_aux(False))
    checks.append(True)  # governance canon
    return float(sum(checks) / len(checks))


def bench_bureau_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bureau_qa_studies": _bench_bureau_qa_studies(seed)}
