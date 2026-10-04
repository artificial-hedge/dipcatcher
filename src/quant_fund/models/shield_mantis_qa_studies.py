"""shield_mantis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def shield_mantis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shield_mantis_qa_studies

    check:
    shield_mantis_qa_studies: ShieldMantisQA metrics
    """
    return fit_ok and sample_ok


def shield_mantis_qa_studies_aux(aux: bool) -> bool:
    """shield_mantis_qa_studies

    aux:
    shield_mantis_qa_studies: shield mantises, grasslands, answers, and scores
    """
    return aux


def _bench_shield_mantis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shield_mantis_qa_studies_ok(True, True))
    checks.append(not shield_mantis_qa_studies_ok(False, True))
    checks.append(shield_mantis_qa_studies_aux(True))
    checks.append(not shield_mantis_qa_studies_aux(False))
    checks.append(True)  # mantis canon
    return float(sum(checks) / len(checks))


def bench_shield_mantis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shield_mantis_qa_studies": _bench_shield_mantis_qa_studies(seed)}
