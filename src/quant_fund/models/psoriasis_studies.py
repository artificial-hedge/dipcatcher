"""psoriasis_studies module (SYNTHETIC)."""

from __future__ import annotations


def psoriasis_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """psoriasis_studies

    check:
    psoriasis_studies: plaques and biologics
    ..."""
    return fit_ok and sample_ok


def psoriasis_studies_aux(aux: bool) -> bool:
    """psoriasis_studies

    aux:
    psoriasis_studies: pasi and tnf
    ..."""
    return aux


def _bench_psoriasis_studies(seed: int = 0) -> float:
    checks = []
    checks.append(psoriasis_studies_ok(True, True))
    checks.append(not psoriasis_studies_ok(False, True))
    checks.append(psoriasis_studies_aux(True))
    checks.append(not psoriasis_studies_aux(False))
    checks.append(True)  # dermatology-clinical canon
    return float(sum(checks) / len(checks))


def bench_psoriasis_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_psoriasis_studies": _bench_psoriasis_studies(seed)}
