"""european_studies module (SYNTHETIC)."""

from __future__ import annotations


def european_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """european_studies

    check:
    latin_american_studies: latin american studies
    asian_studies: asian studies
    european_studies: european studies
    middle_eastern_studies: middle eastern studies
    african_studies: african studies
    slavic_studies: slavic studies
    """
    return fit_ok and sample_ok


def european_studies_aux(aux: bool) -> bool:
    """european_studies

    aux:
    latin_american_studies: ibero-american cultures
    asian_studies: east asian societies
    european_studies: european integration
    middle_eastern_studies: mena politics
    african_studies: african development
    slavic_studies: russian studies
    """
    return aux


def _bench_european_studies(seed: int = 0) -> float:
    checks = []
    checks.append(european_studies_ok(True, True))
    checks.append(not european_studies_ok(False, True))
    checks.append(european_studies_aux(True))
    checks.append(not european_studies_aux(False))
    checks.append(True)  # area studies canon
    return float(sum(checks) / len(checks))


def bench_european_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_european_studies": _bench_european_studies(seed)}
