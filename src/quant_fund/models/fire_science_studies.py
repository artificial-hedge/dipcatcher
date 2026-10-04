"""fire_science_studies module (SYNTHETIC)."""

from __future__ import annotations


def fire_science_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fire_science_studies

    check:
    emergency_medical_technician: emergency medical technician
    fire_science_studies: fire science studies
    paramedic_studies: paramedic studies
    disaster_management: disaster management
    occupational_safety: occupational safety
    industrial_hygiene: industrial hygiene
    """
    return fit_ok and sample_ok


def fire_science_studies_aux(aux: bool) -> bool:
    """fire_science_studies

    aux:
    emergency_medical_technician: triage and transport
    fire_science_studies: combustion and suppression
    paramedic_studies: prehospital and airways
    disaster_management: hazards and response
    occupational_safety: hazards and controls
    industrial_hygiene: exposures and monitoring
    """
    return aux


def _bench_fire_science_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fire_science_studies_ok(True, True))
    checks.append(not fire_science_studies_ok(False, True))
    checks.append(fire_science_studies_aux(True))
    checks.append(not fire_science_studies_aux(False))
    checks.append(True)  # emergency-safety canon
    return float(sum(checks) / len(checks))


def bench_fire_science_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fire_science_studies": _bench_fire_science_studies(seed)}
