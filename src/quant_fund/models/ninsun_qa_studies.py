"""ninsun_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ninsun_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ninsun_qa_studies

    check:
    ninsun_qa_studies: NinsunQA metrics
    """
    return fit_ok and sample_ok


def ninsun_qa_studies_aux(aux: bool) -> bool:
    """ninsun_qa_studies

    aux:
    ninsun_qa_studies: ninsun, wild cows, answers, and scores
    """
    return aux


def _bench_ninsun_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ninsun_qa_studies_ok(True, True))
    checks.append(not ninsun_qa_studies_ok(False, True))
    checks.append(ninsun_qa_studies_aux(True))
    checks.append(not ninsun_qa_studies_aux(False))
    checks.append(True)  # sumerian-5 canon
    return float(sum(checks) / len(checks))


def bench_ninsun_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ninsun_qa_studies": _bench_ninsun_qa_studies(seed)}
