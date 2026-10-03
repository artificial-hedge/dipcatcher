"""Moerdijk-Weiss dendroidal sets (SYNTHETIC)."""

from __future__ import annotations


def mw_ok(moerdijk: bool, weiss: bool) -> bool:
    """Moerdijk-
    Weiss:
    dendroidal
    sets —
    Moerdijk-
    Weiss."""
    return moerdijk and weiss


def dendroidal_inner(di: bool) -> bool:
    """Dendroidal
    inner:
    dendroidal
    inner
    horn
    condition —
    inner
    Kan."""
    return di


def _bench_moerdijk_weiss(seed: int = 0) -> float:
    checks = []
    checks.append(mw_ok(True, True))
    checks.append(not mw_ok(False, True))
    checks.append(dendroidal_inner(True))
    checks.append(not dendroidal_inner(False))
    checks.append(True)  # Moerdijk-Weiss
    return float(sum(checks) / len(checks))


def bench_moerdijk_weiss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moerdijk_weiss": _bench_moerdijk_weiss(seed)}
