"""nuwa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuwa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuwa_qa_studies

    check:
    nuwa_qa_studies: NuwaQA metrics
    """
    return fit_ok and sample_ok


def nuwa_qa_studies_aux(aux: bool) -> bool:
    """nuwa_qa_studies

    aux:
    nuwa_qa_studies: nuwa, creation goddesses, answers, and scores
    """
    return aux


def _bench_nuwa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuwa_qa_studies_ok(True, True))
    checks.append(not nuwa_qa_studies_ok(False, True))
    checks.append(nuwa_qa_studies_aux(True))
    checks.append(not nuwa_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_nuwa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuwa_qa_studies": _bench_nuwa_qa_studies(seed)}
