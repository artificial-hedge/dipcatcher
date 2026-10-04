"""extraction_attack_studies module (SYNTHETIC)."""

from __future__ import annotations


def extraction_attack_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """extraction_attack_studies

    check:
    extraction_attack_studies: Model-extraction fidelity and budget metrics
    """
    return fit_ok and sample_ok


def extraction_attack_studies_aux(aux: bool) -> bool:
    """extraction_attack_studies

    aux:
    extraction_attack_studies: queries, predictions, fidelity, and budgets
    """
    return aux


def _bench_extraction_attack_studies(seed: int = 0) -> float:
    checks = []
    checks.append(extraction_attack_studies_ok(True, True))
    checks.append(not extraction_attack_studies_ok(False, True))
    checks.append(extraction_attack_studies_aux(True))
    checks.append(not extraction_attack_studies_aux(False))
    checks.append(True)  # privacy-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_extraction_attack_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_extraction_attack_studies": _bench_extraction_attack_studies(seed)}
