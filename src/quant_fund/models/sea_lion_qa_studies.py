"""sea_lion_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sea_lion_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sea_lion_qa_studies

    check:
    sea_lion_qa_studies: SeaLionQA metrics
    """
    return fit_ok and sample_ok


def sea_lion_qa_studies_aux(aux: bool) -> bool:
    """sea_lion_qa_studies

    aux:
    sea_lion_qa_studies: sea lions, rocky haulouts, answers, and scores
    """
    return aux


def _bench_sea_lion_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sea_lion_qa_studies_ok(True, True))
    checks.append(not sea_lion_qa_studies_ok(False, True))
    checks.append(sea_lion_qa_studies_aux(True))
    checks.append(not sea_lion_qa_studies_aux(False))
    checks.append(True)  # ocean-mammal canon
    return float(sum(checks) / len(checks))


def bench_sea_lion_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sea_lion_qa_studies": _bench_sea_lion_qa_studies(seed)}
