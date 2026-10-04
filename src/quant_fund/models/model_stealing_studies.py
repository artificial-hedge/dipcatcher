"""model_stealing_studies module (SYNTHETIC)."""

from __future__ import annotations


def model_stealing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """model_stealing_studies

    check:
    model_stealing_studies: model stealing / knockoff extraction and fidelity
    """
    return fit_ok and sample_ok


def model_stealing_studies_aux(aux: bool) -> bool:
    """model_stealing_studies

    aux:
    model_stealing_studies: query budgets, substitutes, and agreement rates
    """
    return aux


def _bench_model_stealing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(model_stealing_studies_ok(True, True))
    checks.append(not model_stealing_studies_ok(False, True))
    checks.append(model_stealing_studies_aux(True))
    checks.append(not model_stealing_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_model_stealing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_model_stealing_studies": _bench_model_stealing_studies(seed)}
