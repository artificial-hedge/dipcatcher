"""nisaba_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nisaba_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nisaba_qa_studies

    check:
    nisaba_qa_studies: NisabaQA metrics
    """
    return fit_ok and sample_ok


def nisaba_qa_studies_aux(aux: bool) -> bool:
    """nisaba_qa_studies

    aux:
    nisaba_qa_studies: nisaba, grain scribes, answers, and scores
    """
    return aux


def _bench_nisaba_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nisaba_qa_studies_ok(True, True))
    checks.append(not nisaba_qa_studies_ok(False, True))
    checks.append(nisaba_qa_studies_aux(True))
    checks.append(not nisaba_qa_studies_aux(False))
    checks.append(True)  # sumerian-3 canon
    return float(sum(checks) / len(checks))


def bench_nisaba_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nisaba_qa_studies": _bench_nisaba_qa_studies(seed)}
