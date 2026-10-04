"""scalable_oversight_studies module (SYNTHETIC)."""

from __future__ import annotations


def scalable_oversight_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """scalable_oversight_studies

    check:
    scalable_oversight_studies: human supervision of superhuman systems/monitoring and delegation
    """
    return fit_ok and sample_ok


def scalable_oversight_studies_aux(aux: bool) -> bool:
    """scalable_oversight_studies

    aux:
    scalable_oversight_studies: oversight decomposition and verification/coverage and gaps
    """
    return aux


def _bench_scalable_oversight_studies(seed: int = 0) -> float:
    checks = []
    checks.append(scalable_oversight_studies_ok(True, True))
    checks.append(not scalable_oversight_studies_ok(False, True))
    checks.append(scalable_oversight_studies_aux(True))
    checks.append(not scalable_oversight_studies_aux(False))
    checks.append(True)  # scalable-oversight canon
    return float(sum(checks) / len(checks))


def bench_scalable_oversight_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scalable_oversight_studies": _bench_scalable_oversight_studies(seed)}
