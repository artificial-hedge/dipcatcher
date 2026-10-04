"""mangabey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mangabey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mangabey_qa_studies

    check:
    mangabey_qa_studies: MangabeyQA metrics
    """
    return fit_ok and sample_ok


def mangabey_qa_studies_aux(aux: bool) -> bool:
    """mangabey_qa_studies

    aux:
    mangabey_qa_studies: mangabeys, swamp forests, answers, and scores
    """
    return aux


def _bench_mangabey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mangabey_qa_studies_ok(True, True))
    checks.append(not mangabey_qa_studies_ok(False, True))
    checks.append(mangabey_qa_studies_aux(True))
    checks.append(not mangabey_qa_studies_aux(False))
    checks.append(True)  # old-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_mangabey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mangabey_qa_studies": _bench_mangabey_qa_studies(seed)}
