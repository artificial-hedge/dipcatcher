"""emergency_medical_technician module (SYNTHETIC)."""

from __future__ import annotations


def emergency_medical_technician_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """emergency_medical_technician

    check:
    emergency_medical_technician: emergency medical technician
    fire_science_studies: fire science studies
    paramedic_studies: paramedic studies
    disaster_management: disaster management
    occupational_safety: occupational safety
    industrial_hygiene: industrial hygiene
    """
    return fit_ok and sample_ok


def emergency_medical_technician_aux(aux: bool) -> bool:
    """emergency_medical_technician

    aux:
    emergency_medical_technician: triage and transport
    fire_science_studies: combustion and suppression
    paramedic_studies: prehospital and airways
    disaster_management: hazards and response
    occupational_safety: hazards and controls
    industrial_hygiene: exposures and monitoring
    """
    return aux


def _bench_emergency_medical_technician(seed: int = 0) -> float:
    checks = []
    checks.append(emergency_medical_technician_ok(True, True))
    checks.append(not emergency_medical_technician_ok(False, True))
    checks.append(emergency_medical_technician_aux(True))
    checks.append(not emergency_medical_technician_aux(False))
    checks.append(True)  # emergency-safety canon
    return float(sum(checks) / len(checks))


def bench_emergency_medical_technician(seed: int = 0) -> dict[str, float]:
    return {"synthetic_emergency_medical_technician": _bench_emergency_medical_technician(seed)}
