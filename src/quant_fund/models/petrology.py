"""petrology module (SYNTHETIC)."""

from __future__ import annotations


def petrology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """petrology

    check:
    stratigraphy: stratigraphy
    structural_geology: structural geology
    petrology: petrology
    geochemistry: geochemistry
    geochronology: geochronology
    paleontology: paleontology
    """
    return fit_ok and sample_ok


def petrology_aux(aux: bool) -> bool:
    """petrology

    aux:
    stratigraphy: sequence stratigraphy
    structural_geology: fault analysis
    petrology: igneous rocks
    geochemistry: isotope ratios
    geochronology: radiometric dating
    paleontology: fossil record
    """
    return aux


def _bench_petrology(seed: int = 0) -> float:
    checks = []
    checks.append(petrology_ok(True, True))
    checks.append(not petrology_ok(False, True))
    checks.append(petrology_aux(True))
    checks.append(not petrology_aux(False))
    checks.append(True)  # geology canon
    return float(sum(checks) / len(checks))


def bench_petrology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_petrology": _bench_petrology(seed)}
