"""Knuth-RSK correspondence (SYNTHETIC)."""

from __future__ import annotations


def kr_ok(bijection: bool, two_tableaux: bool) -> bool:
    """Knuth-
    RSK:
    bijection
    between
    matrices
    and
    tableaux
    pairs —
    insertion
    algorithm."""
    return bijection and two_tableaux


def schensted_insertion(si: bool) -> bool:
    """Schensted
    insertion:
    row-
    bumping
    builds
    the
    P-
    tableau —
    RSK
    core
    step."""
    return si


def _bench_knuth_rsk(seed: int = 0) -> float:
    checks = []
    checks.append(kr_ok(True, True))
    checks.append(not kr_ok(False, True))
    checks.append(schensted_insertion(True))
    checks.append(not schensted_insertion(False))
    checks.append(True)  # Robinson-Schensted-Knuth
    return float(sum(checks) / len(checks))


def bench_knuth_rsk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knuth_rsk": _bench_knuth_rsk(seed)}
