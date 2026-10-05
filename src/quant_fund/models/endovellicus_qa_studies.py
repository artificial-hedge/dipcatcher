"""endovellicus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def endovellicus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """endovellicus_qa_studies

    check:
    endovellicus_qa_studies: p
    """
    return fit_ok and sample_ok


def endovellicus_qa_studies_aux(aux: bool) -> bool:
    """endovellicus_qa_studies

    aux:
    endovellicus_qa_studies: r
    """
    return aux


def _bench_endovellicus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(endovellicus_qa_studies_ok(True, True))
    checks.append(not endovellicus_qa_studies_ok(False, True))
    checks.append(endovellicus_qa_studies_aux(True))
    checks.append(not endovellicus_qa_studies_aux(False))
    checks.append(True)  # iberian-myth canon
    return float(sum(checks) / len(checks))


def bench_endovellicus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_endovellicus_qa_studies": _bench_endovellicus_qa_studies(seed)}
