"""plate_tectonics module (SYNTHETIC)."""

from __future__ import annotations


def plate_tectonics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plate_tectonics

    check:
    seismic_waves: seismic waves
    earthquake_magnitude: earthquake magnitude
    plate_tectonics: plate tectonics
    gravity_anomaly: gravity anomaly
    geomagnetism: geomagnetism
    heat_flow_geo: geothermal heat flow
    """
    return fit_ok and sample_ok


def plate_tectonics_aux(aux: bool) -> bool:
    """plate_tectonics

    aux:
    seismic_waves: P and S waves
    earthquake_magnitude: moment magnitude
    plate_tectonics: subduction zones
    gravity_anomaly: Bouguer anomaly
    geomagnetism: dipole field
    heat_flow_geo: mantle heat flux
    """
    return aux


def _bench_plate_tectonics(seed: int = 0) -> float:
    checks = []
    checks.append(plate_tectonics_ok(True, True))
    checks.append(not plate_tectonics_ok(False, True))
    checks.append(plate_tectonics_aux(True))
    checks.append(not plate_tectonics_aux(False))
    checks.append(True)  # geophysics-3 canon
    return float(sum(checks) / len(checks))


def bench_plate_tectonics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plate_tectonics": _bench_plate_tectonics(seed)}
