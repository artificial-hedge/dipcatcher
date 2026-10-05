"""naa2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def naa2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """naa2_qa_studies

    check:
    naa2_qa_studies: Naa2QA metrics
    """
    return fit_ok and sample_ok


def naa2_qa_studies_aux(aux: bool) -> bool:
    """naa2_qa_studies

    aux:
    naa2_qa_studies: naa2, earth masters, answers, and scores
    """
    return aux


def _bench_naa2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(naa2_qa_studies_ok(True, True))
    checks.append(not naa2_qa_studies_ok(False, True))
    checks.append(naa2_qa_studies_aux(True))
    checks.append(not naa2_qa_studies_aux(False))
    checks.append(True)  # siberian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_naa2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_naa2_qa_studies": _bench_naa2_qa_studies(seed)}
