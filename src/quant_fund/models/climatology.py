"""climatology module (SYNTHETIC)."""

from __future__ import annotations


def climatology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """climatology

    check:
    physical_geography: physical geography
    human_geography: human geography
    cartography: cartography
    remote_sensing: remote sensing
    geomorphology: geomorphology
    climatology: climatology
    """
    return fit_ok and sample_ok


def climatology_aux(aux: bool) -> bool:
    """climatology

    aux:
    physical_geography: earth systems
    human_geography: spatial societies
    cartography: map making
    remote_sensing: satellite imagery
    geomorphology: landform processes
    climatology: climate patterns
    """
    return aux


def _bench_climatology(seed: int = 0) -> float:
    checks = []
    checks.append(climatology_ok(True, True))
    checks.append(not climatology_ok(False, True))
    checks.append(climatology_aux(True))
    checks.append(not climatology_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_climatology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_climatology": _bench_climatology(seed)}
