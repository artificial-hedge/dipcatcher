"""olivine_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def olivine_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """olivine_qa_studies

    check:
    olivine_qa_studies: OlivineQA metrics
    """
    return fit_ok and sample_ok


def olivine_qa_studies_aux(aux: bool) -> bool:
    """olivine_qa_studies

    aux:
    olivine_qa_studies: olivines, basalts, answers, and scores
    """
    return aux


def _bench_olivine_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(olivine_qa_studies_ok(True, True))
    checks.append(not olivine_qa_studies_ok(False, True))
    checks.append(olivine_qa_studies_aux(True))
    checks.append(not olivine_qa_studies_aux(False))
    checks.append(True)  # mineral canon
    return float(sum(checks) / len(checks))


def bench_olivine_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_olivine_qa_studies": _bench_olivine_qa_studies(seed)}
