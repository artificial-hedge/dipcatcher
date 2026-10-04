"""glacier_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def glacier_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glacier_qa_studies

    check:
    glacier_qa_studies: GlacierQA metrics
    """
    return fit_ok and sample_ok


def glacier_qa_studies_aux(aux: bool) -> bool:
    """glacier_qa_studies

    aux:
    glacier_qa_studies: glaciers, flows, answers, and scores
    """
    return aux


def _bench_glacier_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glacier_qa_studies_ok(True, True))
    checks.append(not glacier_qa_studies_ok(False, True))
    checks.append(glacier_qa_studies_aux(True))
    checks.append(not glacier_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_glacier_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glacier_qa_studies": _bench_glacier_qa_studies(seed)}
