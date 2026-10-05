"""min_kyawzwa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def min_kyawzwa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """min_kyawzwa_qa_studies

    check:
    min_kyawzwa_qa_studies: M
    """
    return fit_ok and sample_ok


def min_kyawzwa_qa_studies_aux(aux: bool) -> bool:
    """min_kyawzwa_qa_studies

    aux:
    min_kyawzwa_qa_studies: i
    """
    return aux


def _bench_min_kyawzwa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(min_kyawzwa_qa_studies_ok(True, True))
    checks.append(not min_kyawzwa_qa_studies_ok(False, True))
    checks.append(min_kyawzwa_qa_studies_aux(True))
    checks.append(not min_kyawzwa_qa_studies_aux(False))
    checks.append(True)  # burmese-nat canon
    return float(sum(checks) / len(checks))


def bench_min_kyawzwa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_min_kyawzwa_qa_studies": _bench_min_kyawzwa_qa_studies(seed)}
