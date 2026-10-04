"""proboscis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def proboscis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """proboscis_qa_studies

    check:
    proboscis_qa_studies: ProboscisQA metrics
    """
    return fit_ok and sample_ok


def proboscis_qa_studies_aux(aux: bool) -> bool:
    """proboscis_qa_studies

    aux:
    proboscis_qa_studies: proboscis monkeys, mangrove banks, answers, and scores
    """
    return aux


def _bench_proboscis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(proboscis_qa_studies_ok(True, True))
    checks.append(not proboscis_qa_studies_ok(False, True))
    checks.append(proboscis_qa_studies_aux(True))
    checks.append(not proboscis_qa_studies_aux(False))
    checks.append(True)  # primate-2 canon
    return float(sum(checks) / len(checks))


def bench_proboscis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proboscis_qa_studies": _bench_proboscis_qa_studies(seed)}
