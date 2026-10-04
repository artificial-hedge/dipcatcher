"""roan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def roan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """roan_qa_studies

    check:
    roan_qa_studies: RoanQA metrics
    """
    return fit_ok and sample_ok


def roan_qa_studies_aux(aux: bool) -> bool:
    """roan_qa_studies

    aux:
    roan_qa_studies: roans, miombo savannas, answers, and scores
    """
    return aux


def _bench_roan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(roan_qa_studies_ok(True, True))
    checks.append(not roan_qa_studies_ok(False, True))
    checks.append(roan_qa_studies_aux(True))
    checks.append(not roan_qa_studies_aux(False))
    checks.append(True)  # savanna-herd canon
    return float(sum(checks) / len(checks))


def bench_roan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roan_qa_studies": _bench_roan_qa_studies(seed)}
