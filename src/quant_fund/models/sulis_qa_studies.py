"""sulis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sulis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sulis_qa_studies

    check:
    sulis_qa_studies: B
    """
    return fit_ok and sample_ok


def sulis_qa_studies_aux(aux: bool) -> bool:
    """sulis_qa_studies

    aux:
    sulis_qa_studies: a
    """
    return aux


def _bench_sulis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sulis_qa_studies_ok(True, True))
    checks.append(not sulis_qa_studies_ok(False, True))
    checks.append(sulis_qa_studies_aux(True))
    checks.append(not sulis_qa_studies_aux(False))
    checks.append(True)  # romano-british-myth canon
    return float(sum(checks) / len(checks))


def bench_sulis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sulis_qa_studies": _bench_sulis_qa_studies(seed)}
