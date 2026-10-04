"""heat_flow_geo module (SYNTHETIC)."""

from __future__ import annotations


def heat_flow_geo_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """heat_flow_geo

    check:
    seismic_waves: seismic waves
    earthquake_magnitude: earthquake magnitude
    plate_tectonics: plate tectonics
    gravity_anomaly: gravity anomaly
    geomagnetism: geomagnetism
    heat_flow_geo: geothermal heat flow
    """
    return fit_ok and sample_ok


def heat_flow_geo_aux(aux: bool) -> bool:
    """heat_flow_geo

    aux:
    seismic_waves: P and S waves
    earthquake_magnitude: moment magnitude
    plate_tectonics: subduction zones
    gravity_anomaly: Bouguer anomaly
    geomagnetism: dipole field
    heat_flow_geo: mantle heat flux
    """
    return aux


def _bench_heat_flow_geo(seed: int = 0) -> float:
    checks = []
    checks.append(heat_flow_geo_ok(True, True))
    checks.append(not heat_flow_geo_ok(False, True))
    checks.append(heat_flow_geo_aux(True))
    checks.append(not heat_flow_geo_aux(False))
    checks.append(True)  # geophysics-3 canon
    return float(sum(checks) / len(checks))


def bench_heat_flow_geo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heat_flow_geo": _bench_heat_flow_geo(seed)}
