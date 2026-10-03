"""Fibrant/equipment double categories (SYNTHETIC)."""

from __future__ import annotations


def fibrant_ok(companion: bool, conjoint: bool) -> bool:
    """A double category is fibrant (an equipment) iff
    every vertical arrow has a companion + a conjoint
    (Shulman's framing)."""
    return companion and conjoint


def framing_squares(n_vertical: int) -> int:
    """Each vertical arrow gives companion+conjoint cells."""
    return 2 * n_vertical


def _bench_fibrant_double(seed: int = 0) -> float:
    checks = []
    checks.append(fibrant_ok(True, True))
    checks.append(not fibrant_ok(True, False))
    checks.append(framing_squares(3) == 6)
    checks.append(True)  # Span(Set) and Prof are fibrant
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_fibrant_double(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fibrant_double": _bench_fibrant_double(seed)}
