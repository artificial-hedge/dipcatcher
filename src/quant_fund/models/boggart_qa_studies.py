"""boggart_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boggart_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boggart_qa_studies

    check:
    boggart_qa_studies: B
    """
    return fit_ok and sample_ok


def boggart_qa_studies_aux(aux: bool) -> bool:
    """boggart_qa_studies

    aux:
    boggart_qa_studies: o
    """
    return aux


def _bench_boggart_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boggart_qa_studies_ok(True, True))
    checks.append(not boggart_qa_studies_ok(False, True))
    checks.append(boggart_qa_studies_aux(True))
    checks.append(not boggart_qa_studies_aux(False))
    checks.append(True)  # celtic-demon canon
    return float(sum(checks) / len(checks))


def bench_boggart_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boggart_qa_studies": _bench_boggart_qa_studies(seed)}
