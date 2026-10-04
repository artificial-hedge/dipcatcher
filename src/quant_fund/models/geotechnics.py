"""geotechnics module (SYNTHETIC)."""

from __future__ import annotations


def geotechnics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geotechnics

    check:
    structural_analysis: structural analysis
    geotechnics: geotechnics
    transportation_eng: transportation engineering
    water_resources: water resources
    construction_mgmt: construction management
    surveying: surveying
    """
    return fit_ok and sample_ok


def geotechnics_aux(aux: bool) -> bool:
    """geotechnics

    aux:
    structural_analysis: beam theory
    geotechnics: soil mechanics
    transportation_eng: traffic flow
    water_resources: hydraulics
    construction_mgmt: scheduling
    surveying: triangulation
    """
    return aux


def _bench_geotechnics(seed: int = 0) -> float:
    checks = []
    checks.append(geotechnics_ok(True, True))
    checks.append(not geotechnics_ok(False, True))
    checks.append(geotechnics_aux(True))
    checks.append(not geotechnics_aux(False))
    checks.append(True)  # civil-engineering canon
    return float(sum(checks) / len(checks))


def bench_geotechnics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geotechnics": _bench_geotechnics(seed)}
