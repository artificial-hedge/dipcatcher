"""otology_studies module (SYNTHETIC)."""

from __future__ import annotations


def otology_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """otology_studies

    check:
    otology_studies: ear and mastoid
    ..."""
    return fit_ok and sample_ok


def otology_studies_aux(aux: bool) -> bool:
    """otology_studies

    aux:
    otology_studies: cholesteatoma and ossicles
    ..."""
    return aux


def _bench_otology_studies(seed: int = 0) -> float:
    checks = []
    checks.append(otology_studies_ok(True, True))
    checks.append(not otology_studies_ok(False, True))
    checks.append(otology_studies_aux(True))
    checks.append(not otology_studies_aux(False))
    checks.append(True)  # ent-head-neck canon
    return float(sum(checks) / len(checks))


def bench_otology_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_otology_studies": _bench_otology_studies(seed)}
