"""Telescope conjecture (SYNTHETIC)."""

from __future__ import annotations


def tc_ok2(telescope: bool, smashing: bool) -> bool:
    """Telescope:
    telescope
    conjecture
    at
    chromatic
    levels —
    smashing."""
    return telescope and smashing


def telescope_local(tl: bool) -> bool:
    """Telescope
    local:
    telescopic
    localization
    smashing —
    Bousfield
    local."""
    return tl


def _bench_telescope_conj(seed: int = 0) -> float:
    checks = []
    checks.append(tc_ok2(True, True))
    checks.append(not tc_ok2(False, True))
    checks.append(telescope_local(True))
    checks.append(not telescope_local(False))
    checks.append(True)  # Hopkins-Smith
    return float(sum(checks) / len(checks))


def bench_telescope_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_telescope_conj": _bench_telescope_conj(seed)}
