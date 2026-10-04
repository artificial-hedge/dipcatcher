"""jacamar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jacamar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jacamar_qa_studies

    check:
    jacamar_qa_studies: JacamarQA metrics
    """
    return fit_ok and sample_ok


def jacamar_qa_studies_aux(aux: bool) -> bool:
    """jacamar_qa_studies

    aux:
    jacamar_qa_studies: jacamars, riverbanks, answers, and scores
    """
    return aux


def _bench_jacamar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jacamar_qa_studies_ok(True, True))
    checks.append(not jacamar_qa_studies_ok(False, True))
    checks.append(jacamar_qa_studies_aux(True))
    checks.append(not jacamar_qa_studies_aux(False))
    checks.append(True)  # riverbird canon
    return float(sum(checks) / len(checks))


def bench_jacamar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacamar_qa_studies": _bench_jacamar_qa_studies(seed)}
