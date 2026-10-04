"""booby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def booby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """booby_qa_studies

    check:
    booby_qa_studies: BoobyQA metrics
    """
    return fit_ok and sample_ok


def booby_qa_studies_aux(aux: bool) -> bool:
    """booby_qa_studies

    aux:
    booby_qa_studies: boobies, cliffs, answers, and scores
    """
    return aux


def _bench_booby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(booby_qa_studies_ok(True, True))
    checks.append(not booby_qa_studies_ok(False, True))
    checks.append(booby_qa_studies_aux(True))
    checks.append(not booby_qa_studies_aux(False))
    checks.append(True)  # seabird-3 canon
    return float(sum(checks) / len(checks))


def bench_booby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_booby_qa_studies": _bench_booby_qa_studies(seed)}
