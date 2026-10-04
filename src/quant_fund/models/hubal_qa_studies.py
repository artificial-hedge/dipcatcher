"""hubal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hubal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hubal_qa_studies

    check:
    hubal_qa_studies: o
    """
    return fit_ok and sample_ok


def hubal_qa_studies_aux(aux: bool) -> bool:
    """hubal_qa_studies

    aux:
    hubal_qa_studies: r
    """
    return aux


def _bench_hubal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hubal_qa_studies_ok(True, True))
    checks.append(not hubal_qa_studies_ok(False, True))
    checks.append(hubal_qa_studies_aux(True))
    checks.append(not hubal_qa_studies_aux(False))
    checks.append(True)  # arabian-myth canon
    return float(sum(checks) / len(checks))


def bench_hubal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hubal_qa_studies": _bench_hubal_qa_studies(seed)}
