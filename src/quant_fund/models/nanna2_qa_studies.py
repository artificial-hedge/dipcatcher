"""nanna2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nanna2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nanna2_qa_studies

    check:
    nanna2_qa_studies: Nanna2QA metrics
    """
    return fit_ok and sample_ok


def nanna2_qa_studies_aux(aux: bool) -> bool:
    """nanna2_qa_studies

    aux:
    nanna2_qa_studies: nanna2, moon crescents, answers, and scores
    """
    return aux


def _bench_nanna2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nanna2_qa_studies_ok(True, True))
    checks.append(not nanna2_qa_studies_ok(False, True))
    checks.append(nanna2_qa_studies_aux(True))
    checks.append(not nanna2_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-3 canon
    return float(sum(checks) / len(checks))


def bench_nanna2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nanna2_qa_studies": _bench_nanna2_qa_studies(seed)}
