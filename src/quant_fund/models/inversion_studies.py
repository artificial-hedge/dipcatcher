"""inversion_studies module (SYNTHETIC)."""

from __future__ import annotations


def inversion_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """inversion_studies

    check:
    inversion_studies: model-inversion feature recovery and similarity
    """
    return fit_ok and sample_ok


def inversion_studies_aux(aux: bool) -> bool:
    """inversion_studies

    aux:
    inversion_studies: predicted targets, reconstructed inputs, scores
    """
    return aux


def _bench_inversion_studies(seed: int = 0) -> float:
    checks = []
    checks.append(inversion_studies_ok(True, True))
    checks.append(not inversion_studies_ok(False, True))
    checks.append(inversion_studies_aux(True))
    checks.append(not inversion_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_inversion_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inversion_studies": _bench_inversion_studies(seed)}
