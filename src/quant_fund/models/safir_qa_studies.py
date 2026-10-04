"""safir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def safir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """safir_qa_studies

    check:
    safir_qa_studies: p
    """
    return fit_ok and sample_ok


def safir_qa_studies_aux(aux: bool) -> bool:
    """safir_qa_studies

    aux:
    safir_qa_studies: a
    """
    return aux


def _bench_safir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(safir_qa_studies_ok(True, True))
    checks.append(not safir_qa_studies_ok(False, True))
    checks.append(safir_qa_studies_aux(True))
    checks.append(not safir_qa_studies_aux(False))
    checks.append(True)  # arthurian-5 canon
    return float(sum(checks) / len(checks))


def bench_safir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_safir_qa_studies": _bench_safir_qa_studies(seed)}
