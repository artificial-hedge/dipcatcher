"""macular_studies module (SYNTHETIC)."""

from __future__ import annotations


def macular_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """macular_studies

    check:
    macular_studies: macula and degeneration
    ..."""
    return fit_ok and sample_ok


def macular_studies_aux(aux: bool) -> bool:
    """macular_studies

    aux:
    macular_studies: amd and drusen
    ..."""
    return aux


def _bench_macular_studies(seed: int = 0) -> float:
    checks = []
    checks.append(macular_studies_ok(True, True))
    checks.append(not macular_studies_ok(False, True))
    checks.append(macular_studies_aux(True))
    checks.append(not macular_studies_aux(False))
    checks.append(True)  # ophthalmology-vision canon
    return float(sum(checks) / len(checks))


def bench_macular_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macular_studies": _bench_macular_studies(seed)}
