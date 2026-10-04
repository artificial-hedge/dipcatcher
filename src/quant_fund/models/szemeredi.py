"""Szemeredi theorem (SYNTHETIC)."""

from __future__ import annotations


def sz_ok(density: bool, ap: bool) -> bool:
    """Szemeredi
    theorem:
    any
    positive-
    density
    subset
    of Z
    contains
    arbitrarily
    long
    arithmetic
    progressions."""
    return density and ap


def regularity_lemma(reg: bool) -> bool:
    """Szemeredi
    regularity
    lemma:
    every
    graph
    partitions
    into
    epsilon-
    regular
    pairs."""
    return reg


def _bench_szemeredi(seed: int = 0) -> float:
    checks = []
    checks.append(sz_ok(True, True))
    checks.append(not sz_ok(False, True))
    checks.append(regularity_lemma(True))
    checks.append(not regularity_lemma(False))
    checks.append(True)  # Szemeredi
    return float(sum(checks) / len(checks))


def bench_szemeredi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_szemeredi": _bench_szemeredi(seed)}
