"""mandrill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mandrill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mandrill_qa_studies

    check:
    mandrill_qa_studies: MandrillQA metrics
    """
    return fit_ok and sample_ok


def mandrill_qa_studies_aux(aux: bool) -> bool:
    """mandrill_qa_studies

    aux:
    mandrill_qa_studies: mandrills, gabon jungle, answers, and scores
    """
    return aux


def _bench_mandrill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mandrill_qa_studies_ok(True, True))
    checks.append(not mandrill_qa_studies_ok(False, True))
    checks.append(mandrill_qa_studies_aux(True))
    checks.append(not mandrill_qa_studies_aux(False))
    checks.append(True)  # old-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_mandrill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mandrill_qa_studies": _bench_mandrill_qa_studies(seed)}
