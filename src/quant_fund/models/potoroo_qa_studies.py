"""potoroo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def potoroo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """potoroo_qa_studies

    check:
    potoroo_qa_studies: PotorooQA metrics
    """
    return fit_ok and sample_ok


def potoroo_qa_studies_aux(aux: bool) -> bool:
    """potoroo_qa_studies

    aux:
    potoroo_qa_studies: potoroos, undergrowth, answers, and scores
    """
    return aux


def _bench_potoroo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(potoroo_qa_studies_ok(True, True))
    checks.append(not potoroo_qa_studies_ok(False, True))
    checks.append(potoroo_qa_studies_aux(True))
    checks.append(not potoroo_qa_studies_aux(False))
    checks.append(True)  # marsupial-3 canon
    return float(sum(checks) / len(checks))


def bench_potoroo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_potoroo_qa_studies": _bench_potoroo_qa_studies(seed)}
