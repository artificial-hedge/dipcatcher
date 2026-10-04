"""surveying module (SYNTHETIC)."""

from __future__ import annotations


def surveying_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """surveying

    check:
    structural_analysis: structural analysis
    geotechnics: geotechnics
    transportation_eng: transportation engineering
    water_resources: water resources
    construction_mgmt: construction management
    surveying: surveying
    """
    return fit_ok and sample_ok


def surveying_aux(aux: bool) -> bool:
    """surveying

    aux:
    structural_analysis: beam theory
    geotechnics: soil mechanics
    transportation_eng: traffic flow
    water_resources: hydraulics
    construction_mgmt: scheduling
    surveying: triangulation
    """
    return aux


def _bench_surveying(seed: int = 0) -> float:
    checks = []
    checks.append(surveying_ok(True, True))
    checks.append(not surveying_ok(False, True))
    checks.append(surveying_aux(True))
    checks.append(not surveying_aux(False))
    checks.append(True)  # civil-engineering canon
    return float(sum(checks) / len(checks))


def bench_surveying(seed: int = 0) -> dict[str, float]:
    return {"synthetic_surveying": _bench_surveying(seed)}
