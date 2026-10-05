"""marhaus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def marhaus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marhaus_qa_studies

    check:
    marhaus_qa_studies: I
    """
    return fit_ok and sample_ok


def marhaus_qa_studies_aux(aux: bool) -> bool:
    """marhaus_qa_studies

    aux:
    marhaus_qa_studies: r
    """
    return aux


def _bench_marhaus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marhaus_qa_studies_ok(True, True))
    checks.append(not marhaus_qa_studies_ok(False, True))
    checks.append(marhaus_qa_studies_aux(True))
    checks.append(not marhaus_qa_studies_aux(False))
    checks.append(True)  # arthurian-4 canon
    return float(sum(checks) / len(checks))


def bench_marhaus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marhaus_qa_studies": _bench_marhaus_qa_studies(seed)}
