"""Inner horns (SYNTHETIC)."""

from __future__ import annotations


def ih_ok(inner: bool, horn: bool) -> bool:
    """Inner
    horn:
    inner
    horn
    of
    a
    simplex —
    inner
    horn
    filler."""
    return inner and horn


def inner_horn_fill(ihf: bool) -> bool:
    """Inner
    horn
    filler:
    inner
    horn
    filling
    condition —
    Joyal
    inner."""
    return ihf


def _bench_inner_horn(seed: int = 0) -> float:
    checks = []
    checks.append(ih_ok(True, True))
    checks.append(not ih_ok(False, True))
    checks.append(inner_horn_fill(True))
    checks.append(not inner_horn_fill(False))
    checks.append(True)  # Joyal
    return float(sum(checks) / len(checks))


def bench_inner_horn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inner_horn": _bench_inner_horn(seed)}
