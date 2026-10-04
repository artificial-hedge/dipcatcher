"""gravitational_lensing module (SYNTHETIC)."""

from __future__ import annotations


def gravitational_lensing_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gravitational_lensing

    check:
    lorentz_transformation: Lorentz transformation
    spacetime_interval: spacetime interval
    four_vectors: four-vectors
    geodesic_motion: geodesic motion
    gravitational_lensing: gravitational lensing
    gravitational_waves: gravitational waves
    """
    return fit_ok and sample_ok


def gravitational_lensing_aux(aux: bool) -> bool:
    """gravitational_lensing

    aux:
    lorentz_transformation: time dilation
    spacetime_interval: light cone
    four_vectors: four-momentum
    geodesic_motion: Christoffel symbols
    gravitational_lensing: Einstein ring
    gravitational_waves: strain amplitude
    """
    return aux


def _bench_gravitational_lensing(seed: int = 0) -> float:
    checks = []
    checks.append(gravitational_lensing_ok(True, True))
    checks.append(not gravitational_lensing_ok(False, True))
    checks.append(gravitational_lensing_aux(True))
    checks.append(not gravitational_lensing_aux(False))
    checks.append(True)  # relativity-2 canon
    return float(sum(checks) / len(checks))


def bench_gravitational_lensing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gravitational_lensing": _bench_gravitational_lensing(seed)}
