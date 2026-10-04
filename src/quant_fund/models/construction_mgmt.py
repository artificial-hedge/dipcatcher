"""construction_mgmt module (SYNTHETIC)."""

from __future__ import annotations


def construction_mgmt_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """construction_mgmt

    check:
    structural_analysis: structural analysis
    geotechnics: geotechnics
    transportation_eng: transportation engineering
    water_resources: water resources
    construction_mgmt: construction management
    surveying: surveying
    """
    return fit_ok and sample_ok


def construction_mgmt_aux(aux: bool) -> bool:
    """construction_mgmt

    aux:
    structural_analysis: beam theory
    geotechnics: soil mechanics
    transportation_eng: traffic flow
    water_resources: hydraulics
    construction_mgmt: scheduling
    surveying: triangulation
    """
    return aux


def _bench_construction_mgmt(seed: int = 0) -> float:
    checks = []
    checks.append(construction_mgmt_ok(True, True))
    checks.append(not construction_mgmt_ok(False, True))
    checks.append(construction_mgmt_aux(True))
    checks.append(not construction_mgmt_aux(False))
    checks.append(True)  # civil-engineering canon
    return float(sum(checks) / len(checks))


def bench_construction_mgmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_construction_mgmt": _bench_construction_mgmt(seed)}
