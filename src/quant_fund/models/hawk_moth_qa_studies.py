"""hawk_moth_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hawk_moth_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hawk_moth_qa_studies

    check:
    hawk_moth_qa_studies: HawkMothQA metrics
    """
    return fit_ok and sample_ok


def hawk_moth_qa_studies_aux(aux: bool) -> bool:
    """hawk_moth_qa_studies

    aux:
    hawk_moth_qa_studies: hawk moths, flowers, answers, and scores
    """
    return aux


def _bench_hawk_moth_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hawk_moth_qa_studies_ok(True, True))
    checks.append(not hawk_moth_qa_studies_ok(False, True))
    checks.append(hawk_moth_qa_studies_aux(True))
    checks.append(not hawk_moth_qa_studies_aux(False))
    checks.append(True)  # moth canon
    return float(sum(checks) / len(checks))


def bench_hawk_moth_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hawk_moth_qa_studies": _bench_hawk_moth_qa_studies(seed)}
