"""hornwort_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hornwort_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hornwort_qa_studies

    check:
    hornwort_qa_studies: HornwortQA metrics
    """
    return fit_ok and sample_ok


def hornwort_qa_studies_aux(aux: bool) -> bool:
    """hornwort_qa_studies

    aux:
    hornwort_qa_studies: hornworts, dampfields, answers, and scores
    """
    return aux


def _bench_hornwort_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hornwort_qa_studies_ok(True, True))
    checks.append(not hornwort_qa_studies_ok(False, True))
    checks.append(hornwort_qa_studies_aux(True))
    checks.append(not hornwort_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_hornwort_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hornwort_qa_studies": _bench_hornwort_qa_studies(seed)}
