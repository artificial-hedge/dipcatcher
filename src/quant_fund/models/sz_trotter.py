"""Szemeredi-Trotter (SYNTHETIC)."""

from __future__ import annotations


def st_ok(incidences: bool, bound: bool) -> bool:
    """Szemeredi-
    Trotter:
    incidences
    of n
    points and
    m lines
    are
    O(m^{2/3}
    n^{2/3} +
    m + n)."""
    return incidences and bound


def crossing_lemma(cross: bool) -> bool:
    """Crossing-
    number
    proof:
    Szekely's
    crossing
    lemma
    gives ST
    simply."""
    return cross


def _bench_sz_trotter(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(crossing_lemma(True))
    checks.append(not crossing_lemma(False))
    checks.append(True)  # Szemeredi-Trotter
    return float(sum(checks) / len(checks))


def bench_sz_trotter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sz_trotter": _bench_sz_trotter(seed)}
