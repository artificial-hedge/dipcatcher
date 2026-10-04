"""gargouille_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gargouille_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gargouille_qa_studies

    check:
    gargouille_qa_studies: GargouilleQA metrics
    """
    return fit_ok and sample_ok


def gargouille_qa_studies_aux(aux: bool) -> bool:
    """gargouille_qa_studies

    aux:
    gargouille_qa_studies: gargouilles, seine waters, answers, and scores
    """
    return aux


def _bench_gargouille_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gargouille_qa_studies_ok(True, True))
    checks.append(not gargouille_qa_studies_ok(False, True))
    checks.append(gargouille_qa_studies_aux(True))
    checks.append(not gargouille_qa_studies_aux(False))
    checks.append(True)  # european-beast canon
    return float(sum(checks) / len(checks))


def bench_gargouille_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gargouille_qa_studies": _bench_gargouille_qa_studies(seed)}
