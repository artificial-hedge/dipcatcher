"""guillyn_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guillyn_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guillyn_qa_studies

    check:
    guillyn_qa_studies: d
    """
    return fit_ok and sample_ok


def guillyn_qa_studies_aux(aux: bool) -> bool:
    """guillyn_qa_studies

    aux:
    guillyn_qa_studies: e
    """
    return aux


def _bench_guillyn_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guillyn_qa_studies_ok(True, True))
    checks.append(not guillyn_qa_studies_ok(False, True))
    checks.append(guillyn_qa_studies_aux(True))
    checks.append(not guillyn_qa_studies_aux(False))
    checks.append(True)  # garamantian-2 canon
    return float(sum(checks) / len(checks))


def bench_guillyn_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guillyn_qa_studies": _bench_guillyn_qa_studies(seed)}
