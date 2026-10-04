"""ferret_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ferret_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ferret_qa_studies

    check:
    ferret_qa_studies: FerretQA metrics
    """
    return fit_ok and sample_ok


def ferret_qa_studies_aux(aux: bool) -> bool:
    """ferret_qa_studies

    aux:
    ferret_qa_studies: ferrets, burrows, answers, and scores
    """
    return aux


def _bench_ferret_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ferret_qa_studies_ok(True, True))
    checks.append(not ferret_qa_studies_ok(False, True))
    checks.append(ferret_qa_studies_aux(True))
    checks.append(not ferret_qa_studies_aux(False))
    checks.append(True)  # mammal canon
    return float(sum(checks) / len(checks))


def bench_ferret_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ferret_qa_studies": _bench_ferret_qa_studies(seed)}
