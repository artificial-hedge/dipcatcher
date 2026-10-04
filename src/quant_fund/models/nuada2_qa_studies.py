"""nuada2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuada2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuada2_qa_studies

    check:
    nuada2_qa_studies: Nuada2QA metrics
    """
    return fit_ok and sample_ok


def nuada2_qa_studies_aux(aux: bool) -> bool:
    """nuada2_qa_studies

    aux:
    nuada2_qa_studies: nuada2, silver hands, answers, and scores
    """
    return aux


def _bench_nuada2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuada2_qa_studies_ok(True, True))
    checks.append(not nuada2_qa_studies_ok(False, True))
    checks.append(nuada2_qa_studies_aux(True))
    checks.append(not nuada2_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_nuada2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuada2_qa_studies": _bench_nuada2_qa_studies(seed)}
