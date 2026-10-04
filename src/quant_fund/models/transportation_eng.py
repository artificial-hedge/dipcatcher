"""transportation_eng module (SYNTHETIC)."""

from __future__ import annotations


def transportation_eng_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """transportation_eng

    check:
    structural_analysis: structural analysis
    geotechnics: geotechnics
    transportation_eng: transportation engineering
    water_resources: water resources
    construction_mgmt: construction management
    surveying: surveying
    """
    return fit_ok and sample_ok


def transportation_eng_aux(aux: bool) -> bool:
    """transportation_eng

    aux:
    structural_analysis: beam theory
    geotechnics: soil mechanics
    transportation_eng: traffic flow
    water_resources: hydraulics
    construction_mgmt: scheduling
    surveying: triangulation
    """
    return aux


def _bench_transportation_eng(seed: int = 0) -> float:
    checks = []
    checks.append(transportation_eng_ok(True, True))
    checks.append(not transportation_eng_ok(False, True))
    checks.append(transportation_eng_aux(True))
    checks.append(not transportation_eng_aux(False))
    checks.append(True)  # civil-engineering canon
    return float(sum(checks) / len(checks))


def bench_transportation_eng(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transportation_eng": _bench_transportation_eng(seed)}
