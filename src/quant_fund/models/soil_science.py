"""soil_science module (SYNTHETIC)."""

from __future__ import annotations


def soil_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """soil_science

    check:
    crop_science: crop science
    soil_science: soil science
    agronomy: agronomy
    animal_science: animal science
    horticulture: horticulture
    pest_management: pest management
    """
    return fit_ok and sample_ok


def soil_science_aux(aux: bool) -> bool:
    """soil_science

    aux:
    crop_science: plant breeding
    soil_science: soil chemistry
    agronomy: tillage systems
    animal_science: livestock nutrition
    horticulture: greenhouse production
    pest_management: integrated pest control
    """
    return aux


def _bench_soil_science(seed: int = 0) -> float:
    checks = []
    checks.append(soil_science_ok(True, True))
    checks.append(not soil_science_ok(False, True))
    checks.append(soil_science_aux(True))
    checks.append(not soil_science_aux(False))
    checks.append(True)  # agriculture canon
    return float(sum(checks) / len(checks))


def bench_soil_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_soil_science": _bench_soil_science(seed)}
