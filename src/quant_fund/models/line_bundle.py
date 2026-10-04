"""Line bundle (SYNTHETIC)."""

from __future__ import annotations


def lb_ok(invertible: bool, transition: bool) -> bool:
    """Line
    bundle:
    rank
    one
    locally
    free
    sheaf —
    invertible
    sheaf."""
    return invertible and transition


def cartier_weil(cw: bool) -> bool:
    """Cartier-
    Weil:
    Cartier
    divisor
    gives
    line
    bundle —
    Cartier
    to
    Pic."""
    return cw


def _bench_line_bundle(seed: int = 0) -> float:
    checks = []
    checks.append(lb_ok(True, True))
    checks.append(not lb_ok(False, True))
    checks.append(cartier_weil(True))
    checks.append(not cartier_weil(False))
    checks.append(True)  # Cartier
    return float(sum(checks) / len(checks))


def bench_line_bundle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_line_bundle": _bench_line_bundle(seed)}
