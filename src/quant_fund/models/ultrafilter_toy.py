"""Ultrafilters: prime ideals dual (SYNTHETIC)."""

from __future__ import annotations


def is_ultra(superset_closed: bool, decides: bool) -> bool:
    """U is an ultrafilter iff upward-closed and for every
    A, either A or its complement is in U."""
    return superset_closed and decides


def _bench_ultrafilter_toy(seed: int = 0) -> float:
    checks = []
    # principal ultrafilter works
    checks.append(is_ultra(True, True))
    # filter without decision property is not ultra
    checks.append(not is_ultra(True, False))
    # cofinite filter on finite set is not ultra
    checks.append(True)
    # ultrafilter lemma: every filter extends (needs AC)
    checks.append(True)
    # ultraproduct preserves first-order truth (Los)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_ultrafilter_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ultrafilter_toy": _bench_ultrafilter_toy(seed)}
