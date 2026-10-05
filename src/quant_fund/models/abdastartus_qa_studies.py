"""abdastartus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def abdastartus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """abdastartus_qa_studies

    check:
    abdastartus_qa_studies: s
    """
    return fit_ok and sample_ok


def abdastartus_qa_studies_aux(aux: bool) -> bool:
    """abdastartus_qa_studies

    aux:
    abdastartus_qa_studies: e
    """
    return aux


def _bench_abdastartus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(abdastartus_qa_studies_ok(True, True))
    checks.append(not abdastartus_qa_studies_ok(False, True))
    checks.append(abdastartus_qa_studies_aux(True))
    checks.append(not abdastartus_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_abdastartus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abdastartus_qa_studies": _bench_abdastartus_qa_studies(seed)}
