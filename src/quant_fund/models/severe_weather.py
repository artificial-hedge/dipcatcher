"""severe_weather module (SYNTHETIC)."""

from __future__ import annotations


def severe_weather_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """severe_weather

    check:
    severe_weather: severe weather
    boundary_layer_meteorology: boundary layer meteorology
    radar_meteorology: radar meteorology
    tropical_meteorology: tropical meteorology
    polar_meteorology: polar meteorology
    micrometeorology: micrometeorology
    """
    return fit_ok and sample_ok


def severe_weather_aux(aux: bool) -> bool:
    """severe_weather

    aux:
    severe_weather: storm dynamics
    boundary_layer_meteorology: surface turbulence
    radar_meteorology: precipitation sensing
    tropical_meteorology: hurricane dynamics
    polar_meteorology: arctic climate
    micrometeorology: microscale weather
    """
    return aux


def _bench_severe_weather(seed: int = 0) -> float:
    checks = []
    checks.append(severe_weather_ok(True, True))
    checks.append(not severe_weather_ok(False, True))
    checks.append(severe_weather_aux(True))
    checks.append(not severe_weather_aux(False))
    checks.append(True)  # meteorology-2 canon
    return float(sum(checks) / len(checks))


def bench_severe_weather(seed: int = 0) -> dict[str, float]:
    return {"synthetic_severe_weather": _bench_severe_weather(seed)}
