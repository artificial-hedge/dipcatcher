"""boramei_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boramei_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boramei_qa_studies

    check:
    boramei_qa_studies: B
    """
    return fit_ok and sample_ok


def boramei_qa_studies_aux(aux: bool) -> bool:
    """boramei_qa_studies

    aux:
    boramei_qa_studies: o
    """
    return aux


def _bench_boramei_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boramei_qa_studies_ok(True, True))
    checks.append(not boramei_qa_studies_ok(False, True))
    checks.append(boramei_qa_studies_aux(True))
    checks.append(not boramei_qa_studies_aux(False))
    checks.append(True)  # khmer-demon canon
    return float(sum(checks) / len(checks))


def bench_boramei_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boramei_qa_studies": _bench_boramei_qa_studies(seed)}
