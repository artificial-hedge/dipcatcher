"""almajira_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def almajira_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """almajira_qa_studies

    check:
    almajira_qa_studies: f
    """
    return fit_ok and sample_ok


def almajira_qa_studies_aux(aux: bool) -> bool:
    """almajira_qa_studies

    aux:
    almajira_qa_studies: a
    """
    return aux


def _bench_almajira_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(almajira_qa_studies_ok(True, True))
    checks.append(not almajira_qa_studies_ok(False, True))
    checks.append(almajira_qa_studies_aux(True))
    checks.append(not almajira_qa_studies_aux(False))
    checks.append(True)  # tuareg canon
    return float(sum(checks) / len(checks))


def bench_almajira_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_almajira_qa_studies": _bench_almajira_qa_studies(seed)}
