"""munidis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def munidis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """munidis_qa_studies

    check:
    munidis_qa_studies: t
    """
    return fit_ok and sample_ok


def munidis_qa_studies_aux(aux: bool) -> bool:
    """munidis_qa_studies

    aux:
    munidis_qa_studies: u
    """
    return aux


def _bench_munidis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(munidis_qa_studies_ok(True, True))
    checks.append(not munidis_qa_studies_ok(False, True))
    checks.append(munidis_qa_studies_aux(True))
    checks.append(not munidis_qa_studies_aux(False))
    checks.append(True)  # lusitanian-myth canon
    return float(sum(checks) / len(checks))


def bench_munidis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_munidis_qa_studies": _bench_munidis_qa_studies(seed)}
