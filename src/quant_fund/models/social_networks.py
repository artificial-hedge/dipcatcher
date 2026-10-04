"""social_networks module (SYNTHETIC)."""

from __future__ import annotations


def social_networks_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """social_networks

    check:
    social_networks: social networks
    demography: demography
    criminology: criminology
    urban_sociology: urban sociology
    economic_sociology: economic sociology
    social_stratification: social stratification
    """
    return fit_ok and sample_ok


def social_networks_aux(aux: bool) -> bool:
    """social_networks

    aux:
    social_networks: tie strength
    demography: population dynamics
    criminology: crime patterns
    urban_sociology: urban communities
    economic_sociology: embeddedness
    social_stratification: class mobility
    """
    return aux


def _bench_social_networks(seed: int = 0) -> float:
    checks = []
    checks.append(social_networks_ok(True, True))
    checks.append(not social_networks_ok(False, True))
    checks.append(social_networks_aux(True))
    checks.append(not social_networks_aux(False))
    checks.append(True)  # sociology canon
    return float(sum(checks) / len(checks))


def bench_social_networks(seed: int = 0) -> dict[str, float]:
    return {"synthetic_social_networks": _bench_social_networks(seed)}
