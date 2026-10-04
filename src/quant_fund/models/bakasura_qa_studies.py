"""bakasura_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bakasura_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bakasura_qa_studies

    check:
    bakasura_qa_studies: B
    """
    return fit_ok and sample_ok


def bakasura_qa_studies_aux(aux: bool) -> bool:
    """bakasura_qa_studies

    aux:
    bakasura_qa_studies: a
    """
    return aux


def _bench_bakasura_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bakasura_qa_studies_ok(True, True))
    checks.append(not bakasura_qa_studies_ok(False, True))
    checks.append(bakasura_qa_studies_aux(True))
    checks.append(not bakasura_qa_studies_aux(False))
    checks.append(True)  # hindu-demon canon
    return float(sum(checks) / len(checks))


def bench_bakasura_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bakasura_qa_studies": _bench_bakasura_qa_studies(seed)}
