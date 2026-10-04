"""margay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def margay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """margay_qa_studies

    check:
    margay_qa_studies: MargayQA metrics
    """
    return fit_ok and sample_ok


def margay_qa_studies_aux(aux: bool) -> bool:
    """margay_qa_studies

    aux:
    margay_qa_studies: margays, branches, answers, and scores
    """
    return aux


def _bench_margay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(margay_qa_studies_ok(True, True))
    checks.append(not margay_qa_studies_ok(False, True))
    checks.append(margay_qa_studies_aux(True))
    checks.append(not margay_qa_studies_aux(False))
    checks.append(True)  # wildcat canon
    return float(sum(checks) / len(checks))


def bench_margay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_margay_qa_studies": _bench_margay_qa_studies(seed)}
