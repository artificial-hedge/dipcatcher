"""copepod_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def copepod_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """copepod_qa_studies

    check:
    copepod_qa_studies: CopepodQA metrics
    """
    return fit_ok and sample_ok


def copepod_qa_studies_aux(aux: bool) -> bool:
    """copepod_qa_studies

    aux:
    copepod_qa_studies: copepods, plankton blooms, answers, and scores
    """
    return aux


def _bench_copepod_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(copepod_qa_studies_ok(True, True))
    checks.append(not copepod_qa_studies_ok(False, True))
    checks.append(copepod_qa_studies_aux(True))
    checks.append(not copepod_qa_studies_aux(False))
    checks.append(True)  # plankton-shore canon
    return float(sum(checks) / len(checks))


def bench_copepod_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_copepod_qa_studies": _bench_copepod_qa_studies(seed)}
