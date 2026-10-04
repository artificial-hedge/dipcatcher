"""nyala_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nyala_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nyala_qa_studies

    check:
    nyala_qa_studies: NyalaQA metrics
    """
    return fit_ok and sample_ok


def nyala_qa_studies_aux(aux: bool) -> bool:
    """nyala_qa_studies

    aux:
    nyala_qa_studies: nyalas, thickets, answers, and scores
    """
    return aux


def _bench_nyala_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nyala_qa_studies_ok(True, True))
    checks.append(not nyala_qa_studies_ok(False, True))
    checks.append(nyala_qa_studies_aux(True))
    checks.append(not nyala_qa_studies_aux(False))
    checks.append(True)  # antelope-2 canon
    return float(sum(checks) / len(checks))


def bench_nyala_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nyala_qa_studies": _bench_nyala_qa_studies(seed)}
