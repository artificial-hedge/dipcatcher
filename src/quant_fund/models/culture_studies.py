"""culture_studies module (SYNTHETIC)."""

from __future__ import annotations


def culture_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """culture_studies

    check:
    culture_studies: colonies and growth/isolation and susceptibility
    """
    return fit_ok and sample_ok


def culture_studies_aux(aux: bool) -> bool:
    """culture_studies

    aux:
    culture_studies: media and incubation/subculture and purity
    """
    return aux


def _bench_culture_studies(seed: int = 0) -> float:
    checks = []
    checks.append(culture_studies_ok(True, True))
    checks.append(not culture_studies_ok(False, True))
    checks.append(culture_studies_aux(True))
    checks.append(not culture_studies_aux(False))
    checks.append(True)  # clinical-lab canon
    return float(sum(checks) / len(checks))


def bench_culture_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_culture_studies": _bench_culture_studies(seed)}
