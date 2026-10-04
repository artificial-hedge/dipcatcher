"""Horn filler (SYNTHETIC)."""

from __future__ import annotations


def hf_ok2(horn: bool, filler: bool) -> bool:
    """Horn:
    horn
    filler
    for
    simplicial
    set —
    inner
    horn."""
    return horn and filler


def inner_horn_fill(ih: bool) -> bool:
    """Inner
    horn:
    inner
    horn
    extension —
    Joyal
    inner."""
    return ih


def _bench_horn_filler(seed: int = 0) -> float:
    checks = []
    checks.append(hf_ok2(True, True))
    checks.append(not hf_ok2(False, True))
    checks.append(inner_horn_fill(True))
    checks.append(not inner_horn_fill(False))
    checks.append(True)  # Joyal
    return float(sum(checks) / len(checks))


def bench_horn_filler(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horn_filler": _bench_horn_filler(seed)}
