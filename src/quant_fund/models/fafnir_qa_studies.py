"""fafnir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fafnir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fafnir_qa_studies

    check:
    fafnir_qa_studies: FafnirQA metrics
    """
    return fit_ok and sample_ok


def fafnir_qa_studies_aux(aux: bool) -> bool:
    """fafnir_qa_studies

    aux:
    fafnir_qa_studies: fafnirs, hoard dragons, answers, and scores
    """
    return aux


def _bench_fafnir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fafnir_qa_studies_ok(True, True))
    checks.append(not fafnir_qa_studies_ok(False, True))
    checks.append(fafnir_qa_studies_aux(True))
    checks.append(not fafnir_qa_studies_aux(False))
    checks.append(True)  # norse-warrior canon
    return float(sum(checks) / len(checks))


def bench_fafnir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fafnir_qa_studies": _bench_fafnir_qa_studies(seed)}
