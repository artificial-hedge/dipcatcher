"""radiation_dosimetry module (SYNTHETIC)."""

from __future__ import annotations


def radiation_dosimetry_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """radiation_dosimetry

    check:
    cardiovascular_technology: cardiovascular technology
    nuclear_medicine_technology: nuclear medicine technology
    radiation_dosimetry: radiation dosimetry
    medical_physics_studies: medical physics studies
    dosimetry_studies: dosimetry studies
    radiopharmacy: radiopharmacy
    """
    return fit_ok and sample_ok


def radiation_dosimetry_aux(aux: bool) -> bool:
    """radiation_dosimetry

    aux:
    cardiovascular_technology: hemodynamics and catheters
    nuclear_medicine_technology: isotopes and scanners
    radiation_dosimetry: dose and detectors
    medical_physics_studies: linacs and qa
    dosimetry_studies: tld and calibration
    radiopharmacy: radiotracers and compounding
    """
    return aux


def _bench_radiation_dosimetry(seed: int = 0) -> float:
    checks = []
    checks.append(radiation_dosimetry_ok(True, True))
    checks.append(not radiation_dosimetry_ok(False, True))
    checks.append(radiation_dosimetry_aux(True))
    checks.append(not radiation_dosimetry_aux(False))
    checks.append(True)  # medical-physics canon
    return float(sum(checks) / len(checks))


def bench_radiation_dosimetry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radiation_dosimetry": _bench_radiation_dosimetry(seed)}
