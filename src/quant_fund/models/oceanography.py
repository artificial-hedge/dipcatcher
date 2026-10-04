"""oceanography module (SYNTHETIC)."""

from __future__ import annotations


def oceanography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oceanography

    check:
    oceanography: oceanography
    hydrology_2: hydrology
    seismology: seismology
    glaciology: glaciology
    paleoclimatology: paleoclimatology
    volcanology_2: volcanology
    """
    return fit_ok and sample_ok


def oceanography_aux(aux: bool) -> bool:
    """oceanography

    aux:
    oceanography: currents and cycles
    hydrology_2: water and watersheds
    seismology: waves and faults
    glaciology: ice and flow
    paleoclimatology: proxies and epochs
    volcanology_2: eruptions and magma
    """
    return aux


def _bench_oceanography(seed: int = 0) -> float:
    checks = []
    checks.append(oceanography_ok(True, True))
    checks.append(not oceanography_ok(False, True))
    checks.append(oceanography_aux(True))
    checks.append(not oceanography_aux(False))
    checks.append(True)  # earth-science canon
    return float(sum(checks) / len(checks))


def bench_oceanography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oceanography": _bench_oceanography(seed)}
