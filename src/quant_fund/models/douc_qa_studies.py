"""douc_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def douc_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """douc_qa_studies

    check:
    douc_qa_studies: DoucQA metrics
    """
    return fit_ok and sample_ok


def douc_qa_studies_aux(aux: bool) -> bool:
    """douc_qa_studies

    aux:
    douc_qa_studies: doucs, indochina treelines, answers, and scores
    """
    return aux


def _bench_douc_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(douc_qa_studies_ok(True, True))
    checks.append(not douc_qa_studies_ok(False, True))
    checks.append(douc_qa_studies_aux(True))
    checks.append(not douc_qa_studies_aux(False))
    checks.append(True)  # primate-2 canon
    return float(sum(checks) / len(checks))


def bench_douc_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_douc_qa_studies": _bench_douc_qa_studies(seed)}
