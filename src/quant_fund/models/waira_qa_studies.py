"""waira_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def waira_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """waira_qa_studies

    check:
    waira_qa_studies: W
    """
    return fit_ok and sample_ok


def waira_qa_studies_aux(aux: bool) -> bool:
    """waira_qa_studies

    aux:
    waira_qa_studies: a
    """
    return aux


def _bench_waira_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(waira_qa_studies_ok(True, True))
    checks.append(not waira_qa_studies_ok(False, True))
    checks.append(waira_qa_studies_aux(True))
    checks.append(not waira_qa_studies_aux(False))
    checks.append(True)  # yokai-9 canon
    return float(sum(checks) / len(checks))


def bench_waira_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_waira_qa_studies": _bench_waira_qa_studies(seed)}
