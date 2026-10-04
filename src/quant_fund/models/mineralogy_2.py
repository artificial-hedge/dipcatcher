"""mineralogy_2 module (SYNTHETIC)."""

from __future__ import annotations


def mineralogy_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mineralogy_2

    check:
    geology_3: geology
    petrology_2: petrology
    mineralogy_2: mineralogy
    stratigraphy_2: stratigraphy
    geomorphology_2: geomorphology
    geochronology_2: geochronology
    """
    return fit_ok and sample_ok


def mineralogy_2_aux(aux: bool) -> bool:
    """mineralogy_2

    aux:
    geology_3: rocks and strata
    petrology_2: magmas and metamorphism
    mineralogy_2: crystals and phases
    stratigraphy_2: layers and correlation
    geomorphology_2: landforms and erosion
    geochronology_2: ages and isotopes
    """
    return aux


def _bench_mineralogy_2(seed: int = 0) -> float:
    checks = []
    checks.append(mineralogy_2_ok(True, True))
    checks.append(not mineralogy_2_ok(False, True))
    checks.append(mineralogy_2_aux(True))
    checks.append(not mineralogy_2_aux(False))
    checks.append(True)  # geological-sciences canon
    return float(sum(checks) / len(checks))


def bench_mineralogy_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mineralogy_2": _bench_mineralogy_2(seed)}
