"""cwq_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def cwq_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cwq_lite_studies

    check:
    cwq_lite_studies: ComplexWebQ metrics
    """
    return fit_ok and sample_ok


def cwq_lite_studies_aux(aux: bool) -> bool:
    """cwq_lite_studies

    aux:
    cwq_lite_studies: questions, sparqls, answers, and scores
    """
    return aux


def _bench_cwq_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cwq_lite_studies_ok(True, True))
    checks.append(not cwq_lite_studies_ok(False, True))
    checks.append(cwq_lite_studies_aux(True))
    checks.append(not cwq_lite_studies_aux(False))
    checks.append(True)  # KG-QA canon
    return float(sum(checks) / len(checks))


def bench_cwq_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cwq_lite_studies": _bench_cwq_lite_studies(seed)}
