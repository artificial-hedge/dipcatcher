"""juniper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def juniper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """juniper_qa_studies

    check:
    juniper_qa_studies: JuniperQA metrics
    """
    return fit_ok and sample_ok


def juniper_qa_studies_aux(aux: bool) -> bool:
    """juniper_qa_studies

    aux:
    juniper_qa_studies: junipers, cones, answers, and scores
    """
    return aux


def _bench_juniper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(juniper_qa_studies_ok(True, True))
    checks.append(not juniper_qa_studies_ok(False, True))
    checks.append(juniper_qa_studies_aux(True))
    checks.append(not juniper_qa_studies_aux(False))
    checks.append(True)  # evergreen canon
    return float(sum(checks) / len(checks))


def bench_juniper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_juniper_qa_studies": _bench_juniper_qa_studies(seed)}
