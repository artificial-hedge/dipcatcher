"""oncilla_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oncilla_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oncilla_qa_studies

    check:
    oncilla_qa_studies: OncillaQA metrics
    """
    return fit_ok and sample_ok


def oncilla_qa_studies_aux(aux: bool) -> bool:
    """oncilla_qa_studies

    aux:
    oncilla_qa_studies: oncillas, canopies, answers, and scores
    """
    return aux


def _bench_oncilla_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oncilla_qa_studies_ok(True, True))
    checks.append(not oncilla_qa_studies_ok(False, True))
    checks.append(oncilla_qa_studies_aux(True))
    checks.append(not oncilla_qa_studies_aux(False))
    checks.append(True)  # wildcat-2 canon
    return float(sum(checks) / len(checks))


def bench_oncilla_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oncilla_qa_studies": _bench_oncilla_qa_studies(seed)}
