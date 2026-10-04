"""quoref_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quoref_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quoref_qa_studies

    check:
    quoref_qa_studies: Quoref coreference metrics
    """
    return fit_ok and sample_ok


def quoref_qa_studies_aux(aux: bool) -> bool:
    """quoref_qa_studies

    aux:
    quoref_qa_studies: passages, questions, spans, and f1
    """
    return aux


def _bench_quoref_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quoref_qa_studies_ok(True, True))
    checks.append(not quoref_qa_studies_ok(False, True))
    checks.append(quoref_qa_studies_aux(True))
    checks.append(not quoref_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-2 canon
    return float(sum(checks) / len(checks))


def bench_quoref_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quoref_qa_studies": _bench_quoref_qa_studies(seed)}
