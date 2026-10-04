"""quoll_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quoll_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quoll_qa_studies

    check:
    quoll_qa_studies: QuollQA metrics
    """
    return fit_ok and sample_ok


def quoll_qa_studies_aux(aux: bool) -> bool:
    """quoll_qa_studies

    aux:
    quoll_qa_studies: quolls, dens, answers, and scores
    """
    return aux


def _bench_quoll_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quoll_qa_studies_ok(True, True))
    checks.append(not quoll_qa_studies_ok(False, True))
    checks.append(quoll_qa_studies_aux(True))
    checks.append(not quoll_qa_studies_aux(False))
    checks.append(True)  # marsupial-2 canon
    return float(sum(checks) / len(checks))


def bench_quoll_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quoll_qa_studies": _bench_quoll_qa_studies(seed)}
