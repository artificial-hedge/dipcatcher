"""solder_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def solder_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """solder_qa_studies

    check:
    solder_qa_studies: SolderQA metrics
    """
    return fit_ok and sample_ok


def solder_qa_studies_aux(aux: bool) -> bool:
    """solder_qa_studies

    aux:
    solder_qa_studies: solders, circuits, answers, and scores
    """
    return aux


def _bench_solder_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(solder_qa_studies_ok(True, True))
    checks.append(not solder_qa_studies_ok(False, True))
    checks.append(solder_qa_studies_aux(True))
    checks.append(not solder_qa_studies_aux(False))
    checks.append(True)  # alloy canon
    return float(sum(checks) / len(checks))


def bench_solder_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_solder_qa_studies": _bench_solder_qa_studies(seed)}
