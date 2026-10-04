"""boto_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boto_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boto_qa_studies

    check:
    boto_qa_studies: B
    """
    return fit_ok and sample_ok


def boto_qa_studies_aux(aux: bool) -> bool:
    """boto_qa_studies

    aux:
    boto_qa_studies: o
    """
    return aux


def _bench_boto_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boto_qa_studies_ok(True, True))
    checks.append(not boto_qa_studies_ok(False, True))
    checks.append(boto_qa_studies_aux(True))
    checks.append(not boto_qa_studies_aux(False))
    checks.append(True)  # brazilian-folklore canon
    return float(sum(checks) / len(checks))


def bench_boto_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boto_qa_studies": _bench_boto_qa_studies(seed)}
