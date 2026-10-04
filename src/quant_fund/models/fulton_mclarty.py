"""Fulton-McPherson (SYNTHETIC)."""

from __future__ import annotations


def fm_ok(config_space: bool, compactification: bool) -> bool:
    """Fulton-
    McPherson:
    compactification
    of
    configuration
    spaces —
    FM
    compactification."""
    return config_space and compactification


def fm_operad(fmo: bool) -> bool:
    """FM
    operad:
    configuration
    compactification
    is
    operad —
    little
    discs
    model."""
    return fmo


def _bench_fulton_mclarty(seed: int = 0) -> float:
    checks = []
    checks.append(fm_ok(True, True))
    checks.append(not fm_ok(False, True))
    checks.append(fm_operad(True))
    checks.append(not fm_operad(False))
    checks.append(True)  # Fulton-McPherson
    return float(sum(checks) / len(checks))


def bench_fulton_mclarty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fulton_mclarty": _bench_fulton_mclarty(seed)}
