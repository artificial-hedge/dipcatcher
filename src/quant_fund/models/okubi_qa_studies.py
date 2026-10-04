"""okubi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def okubi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """okubi_qa_studies

    check:
    okubi_qa_studies: OkubiQA metrics
    """
    return fit_ok and sample_ok


def okubi_qa_studies_aux(aux: bool) -> bool:
    """okubi_qa_studies

    aux:
    okubi_qa_studies: okubi, river oracle, answers, and scores
    """
    return aux


def _bench_okubi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(okubi_qa_studies_ok(True, True))
    checks.append(not okubi_qa_studies_ok(False, True))
    checks.append(okubi_qa_studies_aux(True))
    checks.append(not okubi_qa_studies_aux(False))
    checks.append(True)  # african-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_okubi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_okubi_qa_studies": _bench_okubi_qa_studies(seed)}
