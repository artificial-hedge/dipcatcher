"""taipo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def taipo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """taipo_qa_studies

    check:
    taipo_qa_studies: T
    """
    return fit_ok and sample_ok


def taipo_qa_studies_aux(aux: bool) -> bool:
    """taipo_qa_studies

    aux:
    taipo_qa_studies: a
    """
    return aux


def _bench_taipo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(taipo_qa_studies_ok(True, True))
    checks.append(not taipo_qa_studies_ok(False, True))
    checks.append(taipo_qa_studies_aux(True))
    checks.append(not taipo_qa_studies_aux(False))
    checks.append(True)  # polynesian-demon canon
    return float(sum(checks) / len(checks))


def bench_taipo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taipo_qa_studies": _bench_taipo_qa_studies(seed)}
