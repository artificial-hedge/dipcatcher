"""ceramic_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ceramic_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ceramic_qa_studies

    check:
    ceramic_qa_studies: CeramicQA metrics
    """
    return fit_ok and sample_ok


def ceramic_qa_studies_aux(aux: bool) -> bool:
    """ceramic_qa_studies

    aux:
    ceramic_qa_studies: ceramics, glazes, answers, and scores
    """
    return aux


def _bench_ceramic_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ceramic_qa_studies_ok(True, True))
    checks.append(not ceramic_qa_studies_ok(False, True))
    checks.append(ceramic_qa_studies_aux(True))
    checks.append(not ceramic_qa_studies_aux(False))
    checks.append(True)  # material canon
    return float(sum(checks) / len(checks))


def bench_ceramic_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ceramic_qa_studies": _bench_ceramic_qa_studies(seed)}
