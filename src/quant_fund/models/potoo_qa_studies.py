"""potoo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def potoo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """potoo_qa_studies

    check:
    potoo_qa_studies: PotooQA metrics
    """
    return fit_ok and sample_ok


def potoo_qa_studies_aux(aux: bool) -> bool:
    """potoo_qa_studies

    aux:
    potoo_qa_studies: potoos, perches, answers, and scores
    """
    return aux


def _bench_potoo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(potoo_qa_studies_ok(True, True))
    checks.append(not potoo_qa_studies_ok(False, True))
    checks.append(potoo_qa_studies_aux(True))
    checks.append(not potoo_qa_studies_aux(False))
    checks.append(True)  # nightjar-2 canon
    return float(sum(checks) / len(checks))


def bench_potoo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_potoo_qa_studies": _bench_potoo_qa_studies(seed)}
