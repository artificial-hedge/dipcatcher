"""superposition_studies module (SYNTHETIC)."""

from __future__ import annotations


def superposition_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """superposition_studies

    check:
    superposition_studies: toy models and interference/polysemanticity and phase changes
    """
    return fit_ok and sample_ok


def superposition_studies_aux(aux: bool) -> bool:
    """superposition_studies

    aux:
    superposition_studies: feature compression and recovery/dimensions and basis
    """
    return aux


def _bench_superposition_studies(seed: int = 0) -> float:
    checks = []
    checks.append(superposition_studies_ok(True, True))
    checks.append(not superposition_studies_ok(False, True))
    checks.append(superposition_studies_aux(True))
    checks.append(not superposition_studies_aux(False))
    checks.append(True)  # interpretability-3 canon
    return float(sum(checks) / len(checks))


def bench_superposition_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_superposition_studies": _bench_superposition_studies(seed)}
