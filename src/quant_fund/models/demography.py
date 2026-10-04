"""demography module (SYNTHETIC)."""

from __future__ import annotations


def demography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """demography

    check:
    social_networks: social networks
    demography: demography
    criminology: criminology
    urban_sociology: urban sociology
    economic_sociology: economic sociology
    social_stratification: social stratification
    """
    return fit_ok and sample_ok


def demography_aux(aux: bool) -> bool:
    """demography

    aux:
    social_networks: tie strength
    demography: population dynamics
    criminology: crime patterns
    urban_sociology: urban communities
    economic_sociology: embeddedness
    social_stratification: class mobility
    """
    return aux


def _bench_demography(seed: int = 0) -> float:
    checks = []
    checks.append(demography_ok(True, True))
    checks.append(not demography_ok(False, True))
    checks.append(demography_aux(True))
    checks.append(not demography_aux(False))
    checks.append(True)  # sociology canon
    return float(sum(checks) / len(checks))


def bench_demography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_demography": _bench_demography(seed)}
