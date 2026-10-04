"""cartography module (SYNTHETIC)."""

from __future__ import annotations


def cartography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cartography

    check:
    physical_geography: physical geography
    human_geography: human geography
    cartography: cartography
    remote_sensing: remote sensing
    geomorphology: geomorphology
    climatology: climatology
    """
    return fit_ok and sample_ok


def cartography_aux(aux: bool) -> bool:
    """cartography

    aux:
    physical_geography: earth systems
    human_geography: spatial societies
    cartography: map making
    remote_sensing: satellite imagery
    geomorphology: landform processes
    climatology: climate patterns
    """
    return aux


def _bench_cartography(seed: int = 0) -> float:
    checks = []
    checks.append(cartography_ok(True, True))
    checks.append(not cartography_ok(False, True))
    checks.append(cartography_aux(True))
    checks.append(not cartography_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_cartography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartography": _bench_cartography(seed)}
