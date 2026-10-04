"""bontebok_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bontebok_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bontebok_qa_studies

    check:
    bontebok_qa_studies: BontebokQA metrics
    """
    return fit_ok and sample_ok


def bontebok_qa_studies_aux(aux: bool) -> bool:
    """bontebok_qa_studies

    aux:
    bontebok_qa_studies: bonteboks, cape fynbos, answers, and scores
    """
    return aux


def _bench_bontebok_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bontebok_qa_studies_ok(True, True))
    checks.append(not bontebok_qa_studies_ok(False, True))
    checks.append(bontebok_qa_studies_aux(True))
    checks.append(not bontebok_qa_studies_aux(False))
    checks.append(True)  # antelope-3 canon
    return float(sum(checks) / len(checks))


def bench_bontebok_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bontebok_qa_studies": _bench_bontebok_qa_studies(seed)}
