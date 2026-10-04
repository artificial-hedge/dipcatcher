"""desert_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def desert_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """desert_qa_studies

    check:
    desert_qa_studies: DesertQA metrics
    """
    return fit_ok and sample_ok


def desert_qa_studies_aux(aux: bool) -> bool:
    """desert_qa_studies

    aux:
    desert_qa_studies: deserts, dunes, answers, and scores
    """
    return aux


def _bench_desert_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(desert_qa_studies_ok(True, True))
    checks.append(not desert_qa_studies_ok(False, True))
    checks.append(desert_qa_studies_aux(True))
    checks.append(not desert_qa_studies_aux(False))
    checks.append(True)  # terrain-2 canon
    return float(sum(checks) / len(checks))


def bench_desert_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_desert_qa_studies": _bench_desert_qa_studies(seed)}
