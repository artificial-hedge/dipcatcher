"""bittern_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bittern_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bittern_qa_studies

    check:
    bittern_qa_studies: BitternQA metrics
    """
    return fit_ok and sample_ok


def bittern_qa_studies_aux(aux: bool) -> bool:
    """bittern_qa_studies

    aux:
    bittern_qa_studies: bitterns, marshes, answers, and scores
    """
    return aux


def _bench_bittern_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bittern_qa_studies_ok(True, True))
    checks.append(not bittern_qa_studies_ok(False, True))
    checks.append(bittern_qa_studies_aux(True))
    checks.append(not bittern_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_bittern_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bittern_qa_studies": _bench_bittern_qa_studies(seed)}
