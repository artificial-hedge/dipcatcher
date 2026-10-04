"""tanager_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tanager_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tanager_qa_studies

    check:
    tanager_qa_studies: TanagerQA metrics
    """
    return fit_ok and sample_ok


def tanager_qa_studies_aux(aux: bool) -> bool:
    """tanager_qa_studies

    aux:
    tanager_qa_studies: tanagers, forests, answers, and scores
    """
    return aux


def _bench_tanager_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tanager_qa_studies_ok(True, True))
    checks.append(not tanager_qa_studies_ok(False, True))
    checks.append(tanager_qa_studies_aux(True))
    checks.append(not tanager_qa_studies_aux(False))
    checks.append(True)  # songbird-2 canon
    return float(sum(checks) / len(checks))


def bench_tanager_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tanager_qa_studies": _bench_tanager_qa_studies(seed)}
