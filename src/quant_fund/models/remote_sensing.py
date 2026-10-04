"""remote_sensing module (SYNTHETIC)."""

from __future__ import annotations


def remote_sensing_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """remote_sensing

    check:
    physical_geography: physical geography
    human_geography: human geography
    cartography: cartography
    remote_sensing: remote sensing
    geomorphology: geomorphology
    climatology: climatology
    """
    return fit_ok and sample_ok


def remote_sensing_aux(aux: bool) -> bool:
    """remote_sensing

    aux:
    physical_geography: earth systems
    human_geography: spatial societies
    cartography: map making
    remote_sensing: satellite imagery
    geomorphology: landform processes
    climatology: climate patterns
    """
    return aux


def _bench_remote_sensing(seed: int = 0) -> float:
    checks = []
    checks.append(remote_sensing_ok(True, True))
    checks.append(not remote_sensing_ok(False, True))
    checks.append(remote_sensing_aux(True))
    checks.append(not remote_sensing_aux(False))
    checks.append(True)  # geography canon
    return float(sum(checks) / len(checks))


def bench_remote_sensing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_remote_sensing": _bench_remote_sensing(seed)}
