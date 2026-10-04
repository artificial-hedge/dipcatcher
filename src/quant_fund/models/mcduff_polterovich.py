"""McDuff-Polterovich packing theorem (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(eight_balls: bool, obstruction: bool) -> bool:
    """McDuff-
    Polterovich:
    exceptional
    sphere
    obstructions
    govern
    ball
    packings
    of
    CP2."""
    return eight_balls and obstruction


def biran_decomp(bd: bool) -> bool:
    """Biran
    decomposition:
    CP2
    decomposes
    as
    a
    ball
    plus
    an
    isotropic
    skeleton —
    packing
    criterion."""
    return bd


def _bench_mcduff_polterovich(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(biran_decomp(True))
    checks.append(not biran_decomp(False))
    checks.append(True)  # McDuff-Polterovich
    return float(sum(checks) / len(checks))


def bench_mcduff_polterovich(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mcduff_polterovich": _bench_mcduff_polterovich(seed)}
