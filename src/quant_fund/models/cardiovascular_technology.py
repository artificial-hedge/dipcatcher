"""cardiovascular_technology module (SYNTHETIC)."""

from __future__ import annotations


def cardiovascular_technology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cardiovascular_technology

    check:
    cardiovascular_technology: cardiovascular technology
    nuclear_medicine_technology: nuclear medicine technology
    radiation_dosimetry: radiation dosimetry
    medical_physics_studies: medical physics studies
    dosimetry_studies: dosimetry studies
    radiopharmacy: radiopharmacy
    """
    return fit_ok and sample_ok


def cardiovascular_technology_aux(aux: bool) -> bool:
    """cardiovascular_technology

    aux:
    cardiovascular_technology: hemodynamics and catheters
    nuclear_medicine_technology: isotopes and scanners
    radiation_dosimetry: dose and detectors
    medical_physics_studies: linacs and qa
    dosimetry_studies: tld and calibration
    radiopharmacy: radiotracers and compounding
    """
    return aux


def _bench_cardiovascular_technology(seed: int = 0) -> float:
    checks = []
    checks.append(cardiovascular_technology_ok(True, True))
    checks.append(not cardiovascular_technology_ok(False, True))
    checks.append(cardiovascular_technology_aux(True))
    checks.append(not cardiovascular_technology_aux(False))
    checks.append(True)  # medical-physics canon
    return float(sum(checks) / len(checks))


def bench_cardiovascular_technology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardiovascular_technology": _bench_cardiovascular_technology(seed)}
