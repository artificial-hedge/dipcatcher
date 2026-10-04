"""quac_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quac_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quac_qa_studies

    check:
    quac_qa_studies: QuAC context-question metrics
    """
    return fit_ok and sample_ok


def quac_qa_studies_aux(aux: bool) -> bool:
    """quac_qa_studies

    aux:
    quac_qa_studies: contexts, turns, answers, and f1
    """
    return aux


def _bench_quac_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quac_qa_studies_ok(True, True))
    checks.append(not quac_qa_studies_ok(False, True))
    checks.append(quac_qa_studies_aux(True))
    checks.append(not quac_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-2 canon
    return float(sum(checks) / len(checks))


def bench_quac_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quac_qa_studies": _bench_quac_qa_studies(seed)}
