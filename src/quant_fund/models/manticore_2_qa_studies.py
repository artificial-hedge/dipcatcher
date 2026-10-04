"""manticore_2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manticore_2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manticore_2_qa_studies

    check:
    manticore_2_qa_studies: Manticore2QA metrics
    """
    return fit_ok and sample_ok


def manticore_2_qa_studies_aux(aux: bool) -> bool:
    """manticore_2_qa_studies

    aux:
    manticore_2_qa_studies: manticores, crimson dunes, answers, and scores
    """
    return aux


def _bench_manticore_2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manticore_2_qa_studies_ok(True, True))
    checks.append(not manticore_2_qa_studies_ok(False, True))
    checks.append(manticore_2_qa_studies_aux(True))
    checks.append(not manticore_2_qa_studies_aux(False))
    checks.append(True)  # chimera canon
    return float(sum(checks) / len(checks))


def bench_manticore_2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manticore_2_qa_studies": _bench_manticore_2_qa_studies(seed)}
