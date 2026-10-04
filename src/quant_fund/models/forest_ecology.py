"""forest_ecology module (SYNTHETIC)."""

from __future__ import annotations


def forest_ecology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """forest_ecology

    check:
    silviculture: silviculture
    forest_ecology: forest ecology
    timber_harvesting: timber harvesting
    forest_economics: forest economics
    dendrology: dendrology
    wildfire_management: wildfire management
    """
    return fit_ok and sample_ok


def forest_ecology_aux(aux: bool) -> bool:
    """forest_ecology

    aux:
    silviculture: thinning regimes
    forest_ecology: succession dynamics
    timber_harvesting: yield regulation
    forest_economics: Faustmann rotation
    dendrology: tree identification
    wildfire_management: fire behavior
    """
    return aux


def _bench_forest_ecology(seed: int = 0) -> float:
    checks = []
    checks.append(forest_ecology_ok(True, True))
    checks.append(not forest_ecology_ok(False, True))
    checks.append(forest_ecology_aux(True))
    checks.append(not forest_ecology_aux(False))
    checks.append(True)  # forestry canon
    return float(sum(checks) / len(checks))


def bench_forest_ecology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forest_ecology": _bench_forest_ecology(seed)}
