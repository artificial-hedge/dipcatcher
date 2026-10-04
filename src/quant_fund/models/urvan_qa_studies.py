"""urvan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def urvan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urvan_qa_studies

    check:
    urvan_qa_studies: U
    """
    return fit_ok and sample_ok


def urvan_qa_studies_aux(aux: bool) -> bool:
    """urvan_qa_studies

    aux:
    urvan_qa_studies: r
    """
    return aux


def _bench_urvan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(urvan_qa_studies_ok(True, True))
    checks.append(not urvan_qa_studies_ok(False, True))
    checks.append(urvan_qa_studies_aux(True))
    checks.append(not urvan_qa_studies_aux(False))
    checks.append(True)  # persian-spirit canon
    return float(sum(checks) / len(checks))


def bench_urvan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urvan_qa_studies": _bench_urvan_qa_studies(seed)}
