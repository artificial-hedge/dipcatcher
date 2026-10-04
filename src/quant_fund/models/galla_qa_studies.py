"""galla_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def galla_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """galla_qa_studies

    check:
    galla_qa_studies: GallaQA metrics
    """
    return fit_ok and sample_ok


def galla_qa_studies_aux(aux: bool) -> bool:
    """galla_qa_studies

    aux:
    galla_qa_studies: galla, underworld demons, answers, and scores
    """
    return aux


def _bench_galla_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(galla_qa_studies_ok(True, True))
    checks.append(not galla_qa_studies_ok(False, True))
    checks.append(galla_qa_studies_aux(True))
    checks.append(not galla_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_galla_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galla_qa_studies": _bench_galla_qa_studies(seed)}
