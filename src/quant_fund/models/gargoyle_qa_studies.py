"""gargoyle_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gargoyle_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gargoyle_qa_studies

    check:
    gargoyle_qa_studies: GargoyleQA metrics
    """
    return fit_ok and sample_ok


def gargoyle_qa_studies_aux(aux: bool) -> bool:
    """gargoyle_qa_studies

    aux:
    gargoyle_qa_studies: gargoyles, stone waters, answers, and scores
    """
    return aux


def _bench_gargoyle_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gargoyle_qa_studies_ok(True, True))
    checks.append(not gargoyle_qa_studies_ok(False, True))
    checks.append(gargoyle_qa_studies_aux(True))
    checks.append(not gargoyle_qa_studies_aux(False))
    checks.append(True)  # french-beast canon
    return float(sum(checks) / len(checks))


def bench_gargoyle_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gargoyle_qa_studies": _bench_gargoyle_qa_studies(seed)}
