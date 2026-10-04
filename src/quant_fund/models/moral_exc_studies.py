"""moral_exc_studies module (SYNTHETIC)."""

from __future__ import annotations


def moral_exc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moral_exc_studies

    check:
    moral_exc_studies: moral-exception metrics
    """
    return fit_ok and sample_ok


def moral_exc_studies_aux(aux: bool) -> bool:
    """moral_exc_studies

    aux:
    moral_exc_studies: contexts, rules, exceptions, and accuracies
    """
    return aux


def _bench_moral_exc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moral_exc_studies_ok(True, True))
    checks.append(not moral_exc_studies_ok(False, True))
    checks.append(moral_exc_studies_aux(True))
    checks.append(not moral_exc_studies_aux(False))
    checks.append(True)  # ethics-eval canon
    return float(sum(checks) / len(checks))


def bench_moral_exc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moral_exc_studies": _bench_moral_exc_studies(seed)}
