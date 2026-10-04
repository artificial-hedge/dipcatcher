"""qald_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def qald_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qald_lite_studies

    check:
    qald_lite_studies: QALD metrics
    """
    return fit_ok and sample_ok


def qald_lite_studies_aux(aux: bool) -> bool:
    """qald_lite_studies

    aux:
    qald_lite_studies: questions, sparqls, results, and scores
    """
    return aux


def _bench_qald_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qald_lite_studies_ok(True, True))
    checks.append(not qald_lite_studies_ok(False, True))
    checks.append(qald_lite_studies_aux(True))
    checks.append(not qald_lite_studies_aux(False))
    checks.append(True)  # KG-QA canon
    return float(sum(checks) / len(checks))


def bench_qald_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qald_lite_studies": _bench_qald_lite_studies(seed)}
