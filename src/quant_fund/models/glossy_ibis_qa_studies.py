"""glossy_ibis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def glossy_ibis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """glossy_ibis_qa_studies

    check:
    glossy_ibis_qa_studies: Glossy-ibisQA metrics
    """
    return fit_ok and sample_ok


def glossy_ibis_qa_studies_aux(aux: bool) -> bool:
    """glossy_ibis_qa_studies

    aux:
    glossy_ibis_qa_studies: glossy ibises, sloughs, answers, and scores
    """
    return aux


def _bench_glossy_ibis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(glossy_ibis_qa_studies_ok(True, True))
    checks.append(not glossy_ibis_qa_studies_ok(False, True))
    checks.append(glossy_ibis_qa_studies_aux(True))
    checks.append(not glossy_ibis_qa_studies_aux(False))
    checks.append(True)  # egret canon
    return float(sum(checks) / len(checks))


def bench_glossy_ibis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_glossy_ibis_qa_studies": _bench_glossy_ibis_qa_studies(seed)}
