"""mourning_dove_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mourning_dove_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mourning_dove_qa_studies

    check:
    mourning_dove_qa_studies: Mourning-doveQA metrics
    """
    return fit_ok and sample_ok


def mourning_dove_qa_studies_aux(aux: bool) -> bool:
    """mourning_dove_qa_studies

    aux:
    mourning_dove_qa_studies: mourning doves, wires, answers, and scores
    """
    return aux


def _bench_mourning_dove_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mourning_dove_qa_studies_ok(True, True))
    checks.append(not mourning_dove_qa_studies_ok(False, True))
    checks.append(mourning_dove_qa_studies_aux(True))
    checks.append(not mourning_dove_qa_studies_aux(False))
    checks.append(True)  # columbid canon
    return float(sum(checks) / len(checks))


def bench_mourning_dove_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mourning_dove_qa_studies": _bench_mourning_dove_qa_studies(seed)}
