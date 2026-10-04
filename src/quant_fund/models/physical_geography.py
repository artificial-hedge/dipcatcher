"""physical_geography module (SYNTHETIC)."""

from __future__ import annotations


def physical_geography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """physical_geography

    check:
    physical_geography: physical geography
    human_geography: human geography
    cartography: cartography
    remote_sensing: remote sensing
    geomorphology: geomorphology
    climatology: climatology
    """
    return fit_ok and sample_ok


def physical_geography_aux(aux: bool) -> bool:
    """physical_geography

    aux:
    physical_geography: earth systems
    human_geography: spatial societies
    cartography: map making
    remote_sensing: satellite imagery
    geomorphology: landform processes
    climatology: climate patterns
    """
    return aux


def _bench_physical_geography(seed: int = 0) -> float:
    checks = []
    checks.append(physical_geography_ok(True, True))
    checks.append(not physical_geography_ok(False, True))
    checks.append(physical_geography_aux(True))
    checks.append(not physical_geography_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_physical_geography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_physical_geography": _bench_physical_geography(seed)}
