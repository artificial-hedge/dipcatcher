"""prostate_studies module (SYNTHETIC)."""

from __future__ import annotations


def prostate_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prostate_studies

    check:
    prostate_studies: prostate and psa
    ..."""
    return fit_ok and sample_ok


def prostate_studies_aux(aux: bool) -> bool:
    """prostate_studies

    aux:
    prostate_studies: biopsy and gleason
    ..."""
    return aux


def _bench_prostate_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prostate_studies_ok(True, True))
    checks.append(not prostate_studies_ok(False, True))
    checks.append(prostate_studies_aux(True))
    checks.append(not prostate_studies_aux(False))
    checks.append(True)  # urology-andrology canon
    return float(sum(checks) / len(checks))


def bench_prostate_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prostate_studies": _bench_prostate_studies(seed)}
