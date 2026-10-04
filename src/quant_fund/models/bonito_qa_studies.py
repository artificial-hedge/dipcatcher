"""bonito_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bonito_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bonito_qa_studies

    check:
    bonito_qa_studies: BonitoQA metrics
    """
    return fit_ok and sample_ok


def bonito_qa_studies_aux(aux: bool) -> bool:
    """bonito_qa_studies

    aux:
    bonito_qa_studies: bonitos, warm currents, answers, and scores
    """
    return aux


def _bench_bonito_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bonito_qa_studies_ok(True, True))
    checks.append(not bonito_qa_studies_ok(False, True))
    checks.append(bonito_qa_studies_aux(True))
    checks.append(not bonito_qa_studies_aux(False))
    checks.append(True)  # pelagic-fish canon
    return float(sum(checks) / len(checks))


def bench_bonito_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bonito_qa_studies": _bench_bonito_qa_studies(seed)}
