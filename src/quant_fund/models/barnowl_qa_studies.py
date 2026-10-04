"""barnowl_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barnowl_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barnowl_qa_studies

    check:
    barnowl_qa_studies: BarnOwlQA metrics
    """
    return fit_ok and sample_ok


def barnowl_qa_studies_aux(aux: bool) -> bool:
    """barnowl_qa_studies

    aux:
    barnowl_qa_studies: barn owls, barns, answers, and scores
    """
    return aux


def _bench_barnowl_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barnowl_qa_studies_ok(True, True))
    checks.append(not barnowl_qa_studies_ok(False, True))
    checks.append(barnowl_qa_studies_aux(True))
    checks.append(not barnowl_qa_studies_aux(False))
    checks.append(True)  # owl canon
    return float(sum(checks) / len(checks))


def bench_barnowl_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barnowl_qa_studies": _bench_barnowl_qa_studies(seed)}
