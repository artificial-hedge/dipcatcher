"""dosimetry_studies module (SYNTHETIC)."""

from __future__ import annotations


def dosimetry_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dosimetry_studies

    check:
    cardiovascular_technology: cardiovascular technology
    nuclear_medicine_technology: nuclear medicine technology
    radiation_dosimetry: radiation dosimetry
    medical_physics_studies: medical physics studies
    dosimetry_studies: dosimetry studies
    radiopharmacy: radiopharmacy
    """
    return fit_ok and sample_ok


def dosimetry_studies_aux(aux: bool) -> bool:
    """dosimetry_studies

    aux:
    cardiovascular_technology: hemodynamics and catheters
    nuclear_medicine_technology: isotopes and scanners
    radiation_dosimetry: dose and detectors
    medical_physics_studies: linacs and qa
    dosimetry_studies: tld and calibration
    radiopharmacy: radiotracers and compounding
    """
    return aux


def _bench_dosimetry_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dosimetry_studies_ok(True, True))
    checks.append(not dosimetry_studies_ok(False, True))
    checks.append(dosimetry_studies_aux(True))
    checks.append(not dosimetry_studies_aux(False))
    checks.append(True)  # medical-physics canon
    return float(sum(checks) / len(checks))


def bench_dosimetry_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dosimetry_studies": _bench_dosimetry_studies(seed)}
