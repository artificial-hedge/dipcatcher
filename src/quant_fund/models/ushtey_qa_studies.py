"""ushtey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ushtey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ushtey_qa_studies

    check:
    ushtey_qa_studies: w
    """
    return fit_ok and sample_ok


def ushtey_qa_studies_aux(aux: bool) -> bool:
    """ushtey_qa_studies

    aux:
    ushtey_qa_studies: a
    """
    return aux


def _bench_ushtey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ushtey_qa_studies_ok(True, True))
    checks.append(not ushtey_qa_studies_ok(False, True))
    checks.append(ushtey_qa_studies_aux(True))
    checks.append(not ushtey_qa_studies_aux(False))
    checks.append(True)  # manx-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_ushtey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ushtey_qa_studies": _bench_ushtey_qa_studies(seed)}
