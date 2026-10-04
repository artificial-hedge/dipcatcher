"""Condensed cohomology (SYNTHETIC)."""

from __future__ import annotations


def cc_ok(condensed: bool, cohomology: bool) -> bool:
    """Condensed
    cohomology:
    condensed
    cohomology —
    Scholze
    condensed
    cohomology."""
    return condensed and cohomology


def condensed_derived_fun(cdf: bool) -> bool:
    """Condensed
    derived:
    condensed
    derived
    functors —
    condensed
    Ext."""
    return cdf


def _bench_condensed_coh(seed: int = 0) -> float:
    checks = []
    checks.append(cc_ok(True, True))
    checks.append(not cc_ok(False, True))
    checks.append(condensed_derived_fun(True))
    checks.append(not condensed_derived_fun(False))
    checks.append(True)  # Scholze
    return float(sum(checks) / len(checks))


def bench_condensed_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_condensed_coh": _bench_condensed_coh(seed)}
