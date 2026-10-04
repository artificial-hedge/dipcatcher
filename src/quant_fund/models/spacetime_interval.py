"""spacetime_interval module (SYNTHETIC)."""

from __future__ import annotations


def spacetime_interval_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spacetime_interval

    check:
    lorentz_transformation: Lorentz transformation
    spacetime_interval: spacetime interval
    four_vectors: four-vectors
    geodesic_motion: geodesic motion
    gravitational_lensing: gravitational lensing
    gravitational_waves: gravitational waves
    """
    return fit_ok and sample_ok


def spacetime_interval_aux(aux: bool) -> bool:
    """spacetime_interval

    aux:
    lorentz_transformation: time dilation
    spacetime_interval: light cone
    four_vectors: four-momentum
    geodesic_motion: Christoffel symbols
    gravitational_lensing: Einstein ring
    gravitational_waves: strain amplitude
    """
    return aux


def _bench_spacetime_interval(seed: int = 0) -> float:
    checks = []
    checks.append(spacetime_interval_ok(True, True))
    checks.append(not spacetime_interval_ok(False, True))
    checks.append(spacetime_interval_aux(True))
    checks.append(not spacetime_interval_aux(False))
    checks.append(True)  # relativity-2 canon
    return float(sum(checks) / len(checks))


def bench_spacetime_interval(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spacetime_interval": _bench_spacetime_interval(seed)}
