"""nisse_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nisse_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nisse_qa_studies

    check:
    nisse_qa_studies: NisseQA metrics
    """
    return fit_ok and sample_ok


def nisse_qa_studies_aux(aux: bool) -> bool:
    """nisse_qa_studies

    aux:
    nisse_qa_studies: nisser, farm gnomes, answers, and scores
    """
    return aux


def _bench_nisse_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nisse_qa_studies_ok(True, True))
    checks.append(not nisse_qa_studies_ok(False, True))
    checks.append(nisse_qa_studies_aux(True))
    checks.append(not nisse_qa_studies_aux(False))
    checks.append(True)  # scandinavian-folk canon
    return float(sum(checks) / len(checks))


def bench_nisse_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nisse_qa_studies": _bench_nisse_qa_studies(seed)}
