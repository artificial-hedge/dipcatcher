"""ceres_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ceres_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ceres_qa_studies

    check:
    ceres_qa_studies: CeresQA metrics
    """
    return fit_ok and sample_ok


def ceres_qa_studies_aux(aux: bool) -> bool:
    """ceres_qa_studies

    aux:
    ceres_qa_studies: ceres, grain mothers, answers, and scores
    """
    return aux


def _bench_ceres_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ceres_qa_studies_ok(True, True))
    checks.append(not ceres_qa_studies_ok(False, True))
    checks.append(ceres_qa_studies_aux(True))
    checks.append(not ceres_qa_studies_aux(False))
    checks.append(True)  # roman-rural canon
    return float(sum(checks) / len(checks))


def bench_ceres_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ceres_qa_studies": _bench_ceres_qa_studies(seed)}
