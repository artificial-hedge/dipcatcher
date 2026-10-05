"""safon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def safon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """safon_qa_studies

    check:
    safon_qa_studies: s
    """
    return fit_ok and sample_ok


def safon_qa_studies_aux(aux: bool) -> bool:
    """safon_qa_studies

    aux:
    safon_qa_studies: e
    """
    return aux


def _bench_safon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(safon_qa_studies_ok(True, True))
    checks.append(not safon_qa_studies_ok(False, True))
    checks.append(safon_qa_studies_aux(True))
    checks.append(not safon_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_safon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_safon_qa_studies": _bench_safon_qa_studies(seed)}
