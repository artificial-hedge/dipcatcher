"""nuthatch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuthatch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuthatch_qa_studies

    check:
    nuthatch_qa_studies: NuthatchQA metrics
    """
    return fit_ok and sample_ok


def nuthatch_qa_studies_aux(aux: bool) -> bool:
    """nuthatch_qa_studies

    aux:
    nuthatch_qa_studies: nuthatches, trunks, answers, and scores
    """
    return aux


def _bench_nuthatch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuthatch_qa_studies_ok(True, True))
    checks.append(not nuthatch_qa_studies_ok(False, True))
    checks.append(nuthatch_qa_studies_aux(True))
    checks.append(not nuthatch_qa_studies_aux(False))
    checks.append(True)  # songbird-2 canon
    return float(sum(checks) / len(checks))


def bench_nuthatch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuthatch_qa_studies": _bench_nuthatch_qa_studies(seed)}
