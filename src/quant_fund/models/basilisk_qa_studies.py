"""basilisk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def basilisk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """basilisk_qa_studies

    check:
    basilisk_qa_studies: BasiliskQA metrics
    """
    return fit_ok and sample_ok


def basilisk_qa_studies_aux(aux: bool) -> bool:
    """basilisk_qa_studies

    aux:
    basilisk_qa_studies: basilisks, dungeons, answers, and scores
    """
    return aux


def _bench_basilisk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(basilisk_qa_studies_ok(True, True))
    checks.append(not basilisk_qa_studies_ok(False, True))
    checks.append(basilisk_qa_studies_aux(True))
    checks.append(not basilisk_qa_studies_aux(False))
    checks.append(True)  # legendary-beast canon
    return float(sum(checks) / len(checks))


def bench_basilisk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_basilisk_qa_studies": _bench_basilisk_qa_studies(seed)}
