"""mandarin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mandarin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mandarin_qa_studies

    check:
    mandarin_qa_studies: MandarinQA metrics
    """
    return fit_ok and sample_ok


def mandarin_qa_studies_aux(aux: bool) -> bool:
    """mandarin_qa_studies

    aux:
    mandarin_qa_studies: mandarin fish, coral lagoons, answers, and scores
    """
    return aux


def _bench_mandarin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mandarin_qa_studies_ok(True, True))
    checks.append(not mandarin_qa_studies_ok(False, True))
    checks.append(mandarin_qa_studies_aux(True))
    checks.append(not mandarin_qa_studies_aux(False))
    checks.append(True)  # exotic-fauna canon
    return float(sum(checks) / len(checks))


def bench_mandarin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mandarin_qa_studies": _bench_mandarin_qa_studies(seed)}
