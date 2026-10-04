"""condensed_matter_3 module (SYNTHETIC)."""

from __future__ import annotations


def condensed_matter_3_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """condensed_matter_3

    check:
    nanotechnology: nanotechnology
    biophysics_2: biophysics
    condensed_matter_3: condensed matter
    optics_3: optics
    acoustics_2: acoustics
    thermodynamics_2: thermodynamics
    """
    return fit_ok and sample_ok


def condensed_matter_3_aux(aux: bool) -> bool:
    """condensed_matter_3

    aux:
    nanotechnology: nanoscale materials
    biophysics_2: proteins and membranes
    condensed_matter_3: lattices and bands
    optics_3: waves and imaging
    acoustics_2: sound and vibration
    thermodynamics_2: heat and entropy
    """
    return aux


def _bench_condensed_matter_3(seed: int = 0) -> float:
    checks = []
    checks.append(condensed_matter_3_ok(True, True))
    checks.append(not condensed_matter_3_ok(False, True))
    checks.append(condensed_matter_3_aux(True))
    checks.append(not condensed_matter_3_aux(False))
    checks.append(True)  # physics-5 canon
    return float(sum(checks) / len(checks))


def bench_condensed_matter_3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_condensed_matter_3": _bench_condensed_matter_3(seed)}
