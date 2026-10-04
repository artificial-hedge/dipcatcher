"""tuberculosis_studies module (SYNTHETIC)."""

from __future__ import annotations


def tuberculosis_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tuberculosis_studies

    check:
    tuberculosis_studies: mtb and granuloma
    ..."""
    return fit_ok and sample_ok


def tuberculosis_studies_aux(aux: bool) -> bool:
    """tuberculosis_studies

    aux:
    tuberculosis_studies: rif and latent
    ..."""
    return aux


def _bench_tuberculosis_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tuberculosis_studies_ok(True, True))
    checks.append(not tuberculosis_studies_ok(False, True))
    checks.append(tuberculosis_studies_aux(True))
    checks.append(not tuberculosis_studies_aux(False))
    checks.append(True)  # infectious-medicine canon
    return float(sum(checks) / len(checks))


def bench_tuberculosis_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tuberculosis_studies": _bench_tuberculosis_studies(seed)}
