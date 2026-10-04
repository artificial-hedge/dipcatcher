"""nuwa2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuwa2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuwa2_qa_studies

    check:
    nuwa2_qa_studies: Nuwa2QA metrics
    """
    return fit_ok and sample_ok


def nuwa2_qa_studies_aux(aux: bool) -> bool:
    """nuwa2_qa_studies

    aux:
    nuwa2_qa_studies: nuwa2, sky menders, answers, and scores
    """
    return aux


def _bench_nuwa2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuwa2_qa_studies_ok(True, True))
    checks.append(not nuwa2_qa_studies_ok(False, True))
    checks.append(nuwa2_qa_studies_aux(True))
    checks.append(not nuwa2_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_nuwa2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuwa2_qa_studies": _bench_nuwa2_qa_studies(seed)}
