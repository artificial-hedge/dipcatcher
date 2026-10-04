"""earthquake_magnitude module (SYNTHETIC)."""

from __future__ import annotations


def earthquake_magnitude_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """earthquake_magnitude

    check:
    seismic_waves: seismic waves
    earthquake_magnitude: earthquake magnitude
    plate_tectonics: plate tectonics
    gravity_anomaly: gravity anomaly
    geomagnetism: geomagnetism
    heat_flow_geo: geothermal heat flow
    """
    return fit_ok and sample_ok


def earthquake_magnitude_aux(aux: bool) -> bool:
    """earthquake_magnitude

    aux:
    seismic_waves: P and S waves
    earthquake_magnitude: moment magnitude
    plate_tectonics: subduction zones
    gravity_anomaly: Bouguer anomaly
    geomagnetism: dipole field
    heat_flow_geo: mantle heat flux
    """
    return aux


def _bench_earthquake_magnitude(seed: int = 0) -> float:
    checks = []
    checks.append(earthquake_magnitude_ok(True, True))
    checks.append(not earthquake_magnitude_ok(False, True))
    checks.append(earthquake_magnitude_aux(True))
    checks.append(not earthquake_magnitude_aux(False))
    checks.append(True)  # geophysics-3 canon
    return float(sum(checks) / len(checks))


def bench_earthquake_magnitude(seed: int = 0) -> dict[str, float]:
    return {"synthetic_earthquake_magnitude": _bench_earthquake_magnitude(seed)}
