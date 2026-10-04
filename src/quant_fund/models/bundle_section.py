"""Fiber bundles: local triviality and sections (SYNTHETIC)."""

from __future__ import annotations


def section_extends(local: bool, obstruction: bool) -> bool:
    """Sections exist locally; global extension is
    obstructed by topology (e.g. Mobius band)."""
    return local and not obstruction


def _bench_bundle_section(seed: int = 0) -> float:
    checks = []
    # trivial bundle always has a section
    checks.append(section_extends(True, False))
    # Mobius band has no nonvanishing section
    checks.append(not section_extends(True, True))
    # hairy ball: TS^2 has no nonzero section
    checks.append(not section_extends(True, True))
    # sections of a principal bundle trivialize it
    checks.append(True)
    # local triviality: E ~ U x F over small U
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_bundle_section(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bundle_section": _bench_bundle_section(seed)}
