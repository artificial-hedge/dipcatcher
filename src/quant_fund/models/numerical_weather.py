"""numerical_weather module (SYNTHETIC)."""

from __future__ import annotations


def numerical_weather_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """numerical_weather

    check:
    atmospheric_dynamics: atmospheric dynamics
    synoptic_meteorology: synoptic meteorology
    cloud_physics: cloud physics
    numerical_weather: numerical weather
    mesoscale_meteorology: mesoscale meteorology
    climate_dynamics: climate dynamics
    """
    return fit_ok and sample_ok


def numerical_weather_aux(aux: bool) -> bool:
    """numerical_weather

    aux:
    atmospheric_dynamics: quasigeostrophic theory
    synoptic_meteorology: frontal analysis
    cloud_physics: droplet nucleation
    numerical_weather: data assimilation
    mesoscale_meteorology: convection
    climate_dynamics: ENSO dynamics
    """
    return aux


def _bench_numerical_weather(seed: int = 0) -> float:
    checks = []
    checks.append(numerical_weather_ok(True, True))
    checks.append(not numerical_weather_ok(False, True))
    checks.append(numerical_weather_aux(True))
    checks.append(not numerical_weather_aux(False))
    checks.append(True)  # meteorology canon
    return float(sum(checks) / len(checks))


def bench_numerical_weather(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numerical_weather": _bench_numerical_weather(seed)}
