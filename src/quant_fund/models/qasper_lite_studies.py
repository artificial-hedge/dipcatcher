"""qasper_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def qasper_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qasper_lite_studies

    check:
    qasper_lite_studies: Qasper paper-QA metrics
    """
    return fit_ok and sample_ok


def qasper_lite_studies_aux(aux: bool) -> bool:
    """qasper_lite_studies

    aux:
    qasper_lite_studies: papers, questions, answers, and accuracies
    """
    return aux


def _bench_qasper_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qasper_lite_studies_ok(True, True))
    checks.append(not qasper_lite_studies_ok(False, True))
    checks.append(qasper_lite_studies_aux(True))
    checks.append(not qasper_lite_studies_aux(False))
    checks.append(True)  # QA-exotics canon
    return float(sum(checks) / len(checks))


def bench_qasper_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qasper_lite_studies": _bench_qasper_lite_studies(seed)}
