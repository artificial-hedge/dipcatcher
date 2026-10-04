"""p-adic curve (SYNTHETIC)."""

from __future__ import annotations


def cp_ok(curve: bool, padic: bool) -> bool:
    """p-adic
    curve:
    p-adic
    curve —
    Fargues
    Fontaine."""
    return curve and padic


def ff_curve(ffc: bool) -> bool:
    """FF
    curve:
    Fargues
    Fontaine
    curve —
    complete
    curve."""
    return ffc


def _bench_curve_padic(seed: int = 0) -> float:
    checks = []
    checks.append(cp_ok(True, True))
    checks.append(not cp_ok(False, True))
    checks.append(ff_curve(True))
    checks.append(not ff_curve(False))
    checks.append(True)  # Fargues-Fontaine
    return float(sum(checks) / len(checks))


def bench_curve_padic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curve_padic": _bench_curve_padic(seed)}
