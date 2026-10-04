"""locust_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def locust_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """locust_qa_studies

    check:
    locust_qa_studies: LocustQA metrics
    """
    return fit_ok and sample_ok


def locust_qa_studies_aux(aux: bool) -> bool:
    """locust_qa_studies

    aux:
    locust_qa_studies: locusts, swarms, answers, and scores
    """
    return aux


def _bench_locust_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(locust_qa_studies_ok(True, True))
    checks.append(not locust_qa_studies_ok(False, True))
    checks.append(locust_qa_studies_aux(True))
    checks.append(not locust_qa_studies_aux(False))
    checks.append(True)  # insect-2 canon
    return float(sum(checks) / len(checks))


def bench_locust_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_locust_qa_studies": _bench_locust_qa_studies(seed)}
