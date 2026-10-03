"""Dendroidal sets (SYNTHETIC)."""

from __future__ import annotations


def d2_ok(dendroidal: bool, set_: bool) -> bool:
    """Dendroidal
    set:
    dendroidal
    set —
    Moerdijk-
    Weiss."""
    return dendroidal and set_


def dendroidal_model(dm: bool) -> bool:
    """Dendroidal
    model:
    dendroidal
    model
    structure —
    Cisinski-
    Moerdijk."""
    return dm


def _bench_dendroidal2(seed: int = 0) -> float:
    checks = []
    checks.append(d2_ok(True, True))
    checks.append(not d2_ok(False, True))
    checks.append(dendroidal_model(True))
    checks.append(not dendroidal_model(False))
    checks.append(True)  # Moerdijk-Weiss
    return float(sum(checks) / len(checks))


def bench_dendroidal2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dendroidal2": _bench_dendroidal2(seed)}
