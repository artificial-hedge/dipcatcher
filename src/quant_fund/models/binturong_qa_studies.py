"""binturong_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def binturong_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """binturong_qa_studies

    check:
    binturong_qa_studies: BinturongQA metrics
    """
    return fit_ok and sample_ok


def binturong_qa_studies_aux(aux: bool) -> bool:
    """binturong_qa_studies

    aux:
    binturong_qa_studies: binturongs, fig canopies, answers, and scores
    """
    return aux


def _bench_binturong_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(binturong_qa_studies_ok(True, True))
    checks.append(not binturong_qa_studies_ok(False, True))
    checks.append(binturong_qa_studies_aux(True))
    checks.append(not binturong_qa_studies_aux(False))
    checks.append(True)  # mammal canon
    return float(sum(checks) / len(checks))


def bench_binturong_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_binturong_qa_studies": _bench_binturong_qa_studies(seed)}
