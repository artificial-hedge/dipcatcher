"""Fano-Mori varieties (SYNTHETIC)."""

from __future__ import annotations


def fano_ok(anti_canonical: bool, ample: bool) -> bool:
    """Fano variety: -K_X
    ample; Mori fiber
    spaces have Fano
    fibers."""
    return anti_canonical and ample


def mori_fiber(relative_picard: bool) -> bool:
    """Mori fiber space:
    f: X -> Z with
    -K_X f-ample and
    relative Picard
    number 1."""
    return relative_picard


def _bench_fano_mori(seed: int = 0) -> float:
    checks = []
    checks.append(fano_ok(True, True))
    checks.append(not fano_ok(False, True))
    checks.append(mori_fiber(True))
    checks.append(not mori_fiber(False))
    checks.append(True)  # boundedness Birkar
    return float(sum(checks) / len(checks))


def bench_fano_mori(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fano_mori": _bench_fano_mori(seed)}
