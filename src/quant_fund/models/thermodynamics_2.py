"""thermodynamics_2 module (SYNTHETIC)."""

from __future__ import annotations


def thermodynamics_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thermodynamics_2

    check:
    nanotechnology: nanotechnology
    biophysics_2: biophysics
    condensed_matter_3: condensed matter
    optics_3: optics
    acoustics_2: acoustics
    thermodynamics_2: thermodynamics
    """
    return fit_ok and sample_ok


def thermodynamics_2_aux(aux: bool) -> bool:
    """thermodynamics_2

    aux:
    nanotechnology: nanoscale materials
    biophysics_2: proteins and membranes
    condensed_matter_3: lattices and bands
    optics_3: waves and imaging
    acoustics_2: sound and vibration
    thermodynamics_2: heat and entropy
    """
    return aux


def _bench_thermodynamics_2(seed: int = 0) -> float:
    checks = []
    checks.append(thermodynamics_2_ok(True, True))
    checks.append(not thermodynamics_2_ok(False, True))
    checks.append(thermodynamics_2_aux(True))
    checks.append(not thermodynamics_2_aux(False))
    checks.append(True)  # physics-5 canon
    return float(sum(checks) / len(checks))


def bench_thermodynamics_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thermodynamics_2": _bench_thermodynamics_2(seed)}
