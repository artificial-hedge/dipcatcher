"""Fargues-Fontaine curve (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(fontaine: bool, curve: bool) -> bool:
    """Fargues:
    Fargues-
    Fontaine
    curve —
    Fargues-
    Fontaine."""
    return fontaine and curve


def curve_bundle(cb: bool) -> bool:
    """Vector
    bundle:
    vector
    bundles
    on
    the
    Fargues-
    Fontaine
    curve —
    Fargues-
    Fontaine."""
    return cb


def _bench_fontaine_curve(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(curve_bundle(True))
    checks.append(not curve_bundle(False))
    checks.append(True)  # Fargues-Fontaine
    return float(sum(checks) / len(checks))


def bench_fontaine_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fontaine_curve": _bench_fontaine_curve(seed)}
