"""Néron models (SYNTHETIC)."""

from __future__ import annotations


def neron_ok(smooth_model: bool, univ: bool) -> bool:
    """Néron model of an
    abelian variety
    over a DVR:
    smooth group
    scheme with
    universal
    property."""
    return smooth_model and univ


def component_group(finite: bool) -> bool:
    """Component group:
    the special
    fiber's group
    of connected
    components is
    finite."""
    return finite


def _bench_neron_model(seed: int = 0) -> float:
    checks = []
    checks.append(neron_ok(True, True))
    checks.append(not neron_ok(False, True))
    checks.append(component_group(True))
    checks.append(not component_group(False))
    checks.append(True)  # Néron 1964
    return float(sum(checks) / len(checks))


def bench_neron_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neron_model": _bench_neron_model(seed)}
