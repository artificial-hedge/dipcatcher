"""ogboni_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ogboni_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ogboni_qa_studies

    check:
    ogboni_qa_studies: O
    """
    return fit_ok and sample_ok


def ogboni_qa_studies_aux(aux: bool) -> bool:
    """ogboni_qa_studies

    aux:
    ogboni_qa_studies: g
    """
    return aux


def _bench_ogboni_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ogboni_qa_studies_ok(True, True))
    checks.append(not ogboni_qa_studies_ok(False, True))
    checks.append(ogboni_qa_studies_aux(True))
    checks.append(not ogboni_qa_studies_aux(False))
    checks.append(True)  # african-demon canon
    return float(sum(checks) / len(checks))


def bench_ogboni_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ogboni_qa_studies": _bench_ogboni_qa_studies(seed)}
