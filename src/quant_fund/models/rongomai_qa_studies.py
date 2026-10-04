"""rongomai_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rongomai_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rongomai_qa_studies

    check:
    rongomai_qa_studies: RongomaiQA metrics
    """
    return fit_ok and sample_ok


def rongomai_qa_studies_aux(aux: bool) -> bool:
    """rongomai_qa_studies

    aux:
    rongomai_qa_studies: rongomai, whale riders, answers, and scores
    """
    return aux


def _bench_rongomai_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rongomai_qa_studies_ok(True, True))
    checks.append(not rongomai_qa_studies_ok(False, True))
    checks.append(rongomai_qa_studies_aux(True))
    checks.append(not rongomai_qa_studies_aux(False))
    checks.append(True)  # maori-2 canon
    return float(sum(checks) / len(checks))


def bench_rongomai_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rongomai_qa_studies": _bench_rongomai_qa_studies(seed)}
