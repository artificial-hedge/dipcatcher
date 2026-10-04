"""Infinity operads (SYNTHETIC)."""

from __future__ import annotations


def io2_ok(infty: bool, operad: bool) -> bool:
    """Infinity
    operad:
    infinity
    operad —
    Lurie
    infinity
    operad."""
    return infty and operad


def operad_infty_maps(om: bool) -> bool:
    """Operad
    maps:
    operad
    maps
    in
    an
    infinity
    operad —
    higher
    operad."""
    return om


def _bench_infty_operad2(seed: int = 0) -> float:
    checks = []
    checks.append(io2_ok(True, True))
    checks.append(not io2_ok(False, True))
    checks.append(operad_infty_maps(True))
    checks.append(not operad_infty_maps(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_infty_operad2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infty_operad2": _bench_infty_operad2(seed)}
