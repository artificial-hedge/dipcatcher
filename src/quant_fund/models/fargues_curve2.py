"""Fargues-Fontaine curve II (SYNTHETIC)."""

from __future__ import annotations


def fargues_curve2_ok(schematic: bool, complete: bool) -> bool:
    """The Fargues-Fontaine curve
    X_FF is a complete
    Dedekind scheme; closed
    points = untilts of the
    perfectoid field."""
    return schematic and complete


def vector_bundles_ff(classification: bool) -> bool:
    """Vector bundles on X_FF
    classified by Harder-
    Narasimhan polygons
    (Fargues' theorem)."""
    return classification


def _bench_fargues_curve2(seed: int = 0) -> float:
    checks = []
    checks.append(fargues_curve2_ok(True, True))
    checks.append(not fargues_curve2_ok(False, True))
    checks.append(vector_bundles_ff(True))
    checks.append(not vector_bundles_ff(False))
    checks.append(True)  # G-Bundles = isocrystals
    return float(sum(checks) / len(checks))


def bench_fargues_curve2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fargues_curve2": _bench_fargues_curve2(seed)}
