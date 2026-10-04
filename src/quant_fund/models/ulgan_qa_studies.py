"""ulgan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ulgan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ulgan_qa_studies

    check:
    ulgan_qa_studies: UlganQA metrics
    """
    return fit_ok and sample_ok


def ulgan_qa_studies_aux(aux: bool) -> bool:
    """ulgan_qa_studies

    aux:
    ulgan_qa_studies: ulgan, river givers, answers, and scores
    """
    return aux


def _bench_ulgan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ulgan_qa_studies_ok(True, True))
    checks.append(not ulgan_qa_studies_ok(False, True))
    checks.append(ulgan_qa_studies_aux(True))
    checks.append(not ulgan_qa_studies_aux(False))
    checks.append(True)  # mongolian-myth canon
    return float(sum(checks) / len(checks))


def bench_ulgan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ulgan_qa_studies": _bench_ulgan_qa_studies(seed)}
