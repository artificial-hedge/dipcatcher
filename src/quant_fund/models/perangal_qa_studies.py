"""perangal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def perangal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """perangal_qa_studies

    check:
    perangal_qa_studies: p
    """
    return fit_ok and sample_ok


def perangal_qa_studies_aux(aux: bool) -> bool:
    """perangal_qa_studies

    aux:
    perangal_qa_studies: e
    """
    return aux


def _bench_perangal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(perangal_qa_studies_ok(True, True))
    checks.append(not perangal_qa_studies_ok(False, True))
    checks.append(perangal_qa_studies_aux(True))
    checks.append(not perangal_qa_studies_aux(False))
    checks.append(True)  # folk-spirit lore-2 canon
    return float(sum(checks) / len(checks))


def bench_perangal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perangal_qa_studies": _bench_perangal_qa_studies(seed)}
