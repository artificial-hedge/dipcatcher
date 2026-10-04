"""shoveler_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shoveler_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shoveler_qa_studies

    check:
    shoveler_qa_studies: ShovelerQA metrics
    """
    return fit_ok and sample_ok


def shoveler_qa_studies_aux(aux: bool) -> bool:
    """shoveler_qa_studies

    aux:
    shoveler_qa_studies: shovelers, shallows, answers, and scores
    """
    return aux


def _bench_shoveler_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shoveler_qa_studies_ok(True, True))
    checks.append(not shoveler_qa_studies_ok(False, True))
    checks.append(shoveler_qa_studies_aux(True))
    checks.append(not shoveler_qa_studies_aux(False))
    checks.append(True)  # waterfowl canon
    return float(sum(checks) / len(checks))


def bench_shoveler_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shoveler_qa_studies": _bench_shoveler_qa_studies(seed)}
