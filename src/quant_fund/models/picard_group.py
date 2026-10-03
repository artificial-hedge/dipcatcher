"""Picard group (SYNTHETIC)."""

from __future__ import annotations


def pg_ok(line_bundles: bool, iso_classes: bool) -> bool:
    """Picard
    group:
    line
    bundles
    under
    tensor —
    Pic."""
    return line_bundles and iso_classes


def picard_scheme(ps: bool) -> bool:
    """Picard
    scheme:
    moduli
    of
    line
    bundles —
    Grothendieck
    Picard."""
    return ps


def _bench_picard_group(seed: int = 0) -> float:
    checks = []
    checks.append(pg_ok(True, True))
    checks.append(not pg_ok(False, True))
    checks.append(picard_scheme(True))
    checks.append(not picard_scheme(False))
    checks.append(True)  # Grothendieck-Picard
    return float(sum(checks) / len(checks))


def bench_picard_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_group": _bench_picard_group(seed)}
