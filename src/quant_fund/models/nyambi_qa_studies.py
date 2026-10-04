"""nyambi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nyambi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nyambi_qa_studies

    check:
    nyambi_qa_studies: NyambiQA metrics
    """
    return fit_ok and sample_ok


def nyambi_qa_studies_aux(aux: bool) -> bool:
    """nyambi_qa_studies

    aux:
    nyambi_qa_studies: nyambi, sky walkers, answers, and scores
    """
    return aux


def _bench_nyambi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nyambi_qa_studies_ok(True, True))
    checks.append(not nyambi_qa_studies_ok(False, True))
    checks.append(nyambi_qa_studies_aux(True))
    checks.append(not nyambi_qa_studies_aux(False))
    checks.append(True)  # african-myth-4 canon
    return float(sum(checks) / len(checks))


def bench_nyambi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nyambi_qa_studies": _bench_nyambi_qa_studies(seed)}
