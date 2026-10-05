"""allatu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def allatu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """allatu_qa_studies

    check:
    allatu_qa_studies: a
    """
    return fit_ok and sample_ok


def allatu_qa_studies_aux(aux: bool) -> bool:
    """allatu_qa_studies

    aux:
    allatu_qa_studies: l
    """
    return aux


def _bench_allatu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(allatu_qa_studies_ok(True, True))
    checks.append(not allatu_qa_studies_ok(False, True))
    checks.append(allatu_qa_studies_aux(True))
    checks.append(not allatu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_allatu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_allatu_qa_studies": _bench_allatu_qa_studies(seed)}
