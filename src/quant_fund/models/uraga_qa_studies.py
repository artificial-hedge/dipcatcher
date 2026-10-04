"""uraga_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def uraga_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """uraga_qa_studies

    check:
    uraga_qa_studies: UragaQA metrics
    """
    return fit_ok and sample_ok


def uraga_qa_studies_aux(aux: bool) -> bool:
    """uraga_qa_studies

    aux:
    uraga_qa_studies: uragas, serpent folk, answers, and scores
    """
    return aux


def _bench_uraga_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(uraga_qa_studies_ok(True, True))
    checks.append(not uraga_qa_studies_ok(False, True))
    checks.append(uraga_qa_studies_aux(True))
    checks.append(not uraga_qa_studies_aux(False))
    checks.append(True)  # hindu-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_uraga_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uraga_qa_studies": _bench_uraga_qa_studies(seed)}
