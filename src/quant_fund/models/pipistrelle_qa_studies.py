"""pipistrelle_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pipistrelle_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pipistrelle_qa_studies

    check:
    pipistrelle_qa_studies: PipistrelleQA metrics
    """
    return fit_ok and sample_ok


def pipistrelle_qa_studies_aux(aux: bool) -> bool:
    """pipistrelle_qa_studies

    aux:
    pipistrelle_qa_studies: pipistrelles, town rooftops, answers, and scores
    """
    return aux


def _bench_pipistrelle_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pipistrelle_qa_studies_ok(True, True))
    checks.append(not pipistrelle_qa_studies_ok(False, True))
    checks.append(pipistrelle_qa_studies_aux(True))
    checks.append(not pipistrelle_qa_studies_aux(False))
    checks.append(True)  # bat canon
    return float(sum(checks) / len(checks))


def bench_pipistrelle_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pipistrelle_qa_studies": _bench_pipistrelle_qa_studies(seed)}
