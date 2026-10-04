"""stratigraphy module (SYNTHETIC)."""

from __future__ import annotations


def stratigraphy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stratigraphy

    check:
    stratigraphy: stratigraphy
    structural_geology: structural geology
    petrology: petrology
    geochemistry: geochemistry
    geochronology: geochronology
    paleontology: paleontology
    """
    return fit_ok and sample_ok


def stratigraphy_aux(aux: bool) -> bool:
    """stratigraphy

    aux:
    stratigraphy: sequence stratigraphy
    structural_geology: fault analysis
    petrology: igneous rocks
    geochemistry: isotope ratios
    geochronology: radiometric dating
    paleontology: fossil record
    """
    return aux


def _bench_stratigraphy(seed: int = 0) -> float:
    checks = []
    checks.append(stratigraphy_ok(True, True))
    checks.append(not stratigraphy_ok(False, True))
    checks.append(stratigraphy_aux(True))
    checks.append(not stratigraphy_aux(False))
    checks.append(True)  # geology canon
    return float(sum(checks) / len(checks))


def bench_stratigraphy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stratigraphy": _bench_stratigraphy(seed)}
