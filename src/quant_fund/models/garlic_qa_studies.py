"""garlic_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def garlic_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """garlic_qa_studies

    check:
    garlic_qa_studies: GarlicQA metrics
    """
    return fit_ok and sample_ok


def garlic_qa_studies_aux(aux: bool) -> bool:
    """garlic_qa_studies

    aux:
    garlic_qa_studies: garlic, cloves, answers, and scores
    """
    return aux


def _bench_garlic_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(garlic_qa_studies_ok(True, True))
    checks.append(not garlic_qa_studies_ok(False, True))
    checks.append(garlic_qa_studies_aux(True))
    checks.append(not garlic_qa_studies_aux(False))
    checks.append(True)  # vegetable canon
    return float(sum(checks) / len(checks))


def bench_garlic_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_garlic_qa_studies": _bench_garlic_qa_studies(seed)}
