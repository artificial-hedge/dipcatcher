"""western_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def western_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """western_qa_studies

    check:
    western_qa_studies: WesternQA metrics
    """
    return fit_ok and sample_ok


def western_qa_studies_aux(aux: bool) -> bool:
    """western_qa_studies

    aux:
    western_qa_studies: western lemurs, western forests, answers, and scores
    """
    return aux


def _bench_western_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(western_qa_studies_ok(True, True))
    checks.append(not western_qa_studies_ok(False, True))
    checks.append(western_qa_studies_aux(True))
    checks.append(not western_qa_studies_aux(False))
    checks.append(True)  # lemur-region canon
    return float(sum(checks) / len(checks))


def bench_western_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_western_qa_studies": _bench_western_qa_studies(seed)}
