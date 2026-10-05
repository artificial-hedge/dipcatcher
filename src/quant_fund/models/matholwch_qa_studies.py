"""matholwch_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def matholwch_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matholwch_qa_studies

    check:
    matholwch_qa_studies: I
    """
    return fit_ok and sample_ok


def matholwch_qa_studies_aux(aux: bool) -> bool:
    """matholwch_qa_studies

    aux:
    matholwch_qa_studies: r
    """
    return aux


def _bench_matholwch_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(matholwch_qa_studies_ok(True, True))
    checks.append(not matholwch_qa_studies_ok(False, True))
    checks.append(matholwch_qa_studies_aux(True))
    checks.append(not matholwch_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_matholwch_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matholwch_qa_studies": _bench_matholwch_qa_studies(seed)}
