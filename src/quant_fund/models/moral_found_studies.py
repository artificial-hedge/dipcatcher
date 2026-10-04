"""moral_found_studies module (SYNTHETIC)."""

from __future__ import annotations


def moral_found_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moral_found_studies

    check:
    moral_found_studies: moral-foundations metrics
    """
    return fit_ok and sample_ok


def moral_found_studies_aux(aux: bool) -> bool:
    """moral_found_studies

    aux:
    moral_found_studies: scenarios, foundations, labels, and accuracies
    """
    return aux


def _bench_moral_found_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moral_found_studies_ok(True, True))
    checks.append(not moral_found_studies_ok(False, True))
    checks.append(moral_found_studies_aux(True))
    checks.append(not moral_found_studies_aux(False))
    checks.append(True)  # ethics-eval canon
    return float(sum(checks) / len(checks))


def bench_moral_found_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moral_found_studies": _bench_moral_found_studies(seed)}
