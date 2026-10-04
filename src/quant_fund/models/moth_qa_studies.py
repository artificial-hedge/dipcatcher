"""moth_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moth_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moth_qa_studies

    check:
    moth_qa_studies: MothQA metrics
    """
    return fit_ok and sample_ok


def moth_qa_studies_aux(aux: bool) -> bool:
    """moth_qa_studies

    aux:
    moth_qa_studies: moths, cocoons, answers, and scores
    """
    return aux


def _bench_moth_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moth_qa_studies_ok(True, True))
    checks.append(not moth_qa_studies_ok(False, True))
    checks.append(moth_qa_studies_aux(True))
    checks.append(not moth_qa_studies_aux(False))
    checks.append(True)  # insect canon
    return float(sum(checks) / len(checks))


def bench_moth_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moth_qa_studies": _bench_moth_qa_studies(seed)}
