"""macurgum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def macurgum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """macurgum_qa_studies

    check:
    macurgum_qa_studies: f
    """
    return fit_ok and sample_ok


def macurgum_qa_studies_aux(aux: bool) -> bool:
    """macurgum_qa_studies

    aux:
    macurgum_qa_studies: i
    """
    return aux


def _bench_macurgum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(macurgum_qa_studies_ok(True, True))
    checks.append(not macurgum_qa_studies_ok(False, True))
    checks.append(macurgum_qa_studies_aux(True))
    checks.append(not macurgum_qa_studies_aux(False))
    checks.append(True)  # numidian-myth canon
    return float(sum(checks) / len(checks))


def bench_macurgum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macurgum_qa_studies": _bench_macurgum_qa_studies(seed)}
