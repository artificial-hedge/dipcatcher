"""nuada_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nuada_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuada_qa_studies

    check:
    nuada_qa_studies: NuadaQA metrics
    """
    return fit_ok and sample_ok


def nuada_qa_studies_aux(aux: bool) -> bool:
    """nuada_qa_studies

    aux:
    nuada_qa_studies: nuada, silver hands, answers, and scores
    """
    return aux


def _bench_nuada_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nuada_qa_studies_ok(True, True))
    checks.append(not nuada_qa_studies_ok(False, True))
    checks.append(nuada_qa_studies_aux(True))
    checks.append(not nuada_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_nuada_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuada_qa_studies": _bench_nuada_qa_studies(seed)}
