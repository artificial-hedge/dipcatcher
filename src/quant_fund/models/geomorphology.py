"""geomorphology module (SYNTHETIC)."""

from __future__ import annotations


def geomorphology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geomorphology

    check:
    physical_geography: physical geography
    human_geography: human geography
    cartography: cartography
    remote_sensing: remote sensing
    geomorphology: geomorphology
    climatology: climatology
    """
    return fit_ok and sample_ok


def geomorphology_aux(aux: bool) -> bool:
    """geomorphology

    aux:
    physical_geography: earth systems
    human_geography: spatial societies
    cartography: map making
    remote_sensing: satellite imagery
    geomorphology: landform processes
    climatology: climate patterns
    """
    return aux


def _bench_geomorphology(seed: int = 0) -> float:
    checks = []
    checks.append(geomorphology_ok(True, True))
    checks.append(not geomorphology_ok(False, True))
    checks.append(geomorphology_aux(True))
    checks.append(not geomorphology_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_geomorphology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geomorphology": _bench_geomorphology(seed)}
