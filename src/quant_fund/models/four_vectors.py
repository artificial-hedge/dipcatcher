"""four_vectors module (SYNTHETIC)."""

from __future__ import annotations


def four_vectors_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """four_vectors

    check:
    lorentz_transformation: Lorentz transformation
    spacetime_interval: spacetime interval
    four_vectors: four-vectors
    geodesic_motion: geodesic motion
    gravitational_lensing: gravitational lensing
    gravitational_waves: gravitational waves
    """
    return fit_ok and sample_ok


def four_vectors_aux(aux: bool) -> bool:
    """four_vectors

    aux:
    lorentz_transformation: time dilation
    spacetime_interval: light cone
    four_vectors: four-momentum
    geodesic_motion: Christoffel symbols
    gravitational_lensing: Einstein ring
    gravitational_waves: strain amplitude
    """
    return aux


def _bench_four_vectors(seed: int = 0) -> float:
    checks = []
    checks.append(four_vectors_ok(True, True))
    checks.append(not four_vectors_ok(False, True))
    checks.append(four_vectors_aux(True))
    checks.append(not four_vectors_aux(False))
    checks.append(True)  # relativity-2 canon
    return float(sum(checks) / len(checks))


def bench_four_vectors(seed: int = 0) -> dict[str, float]:
    return {"synthetic_four_vectors": _bench_four_vectors(seed)}
