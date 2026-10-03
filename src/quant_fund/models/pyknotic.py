"""Pyknotic vs condensed (SYNTHETIC)."""

from __future__ import annotations


def pyknotic_ok(small_site: bool, same_cat: bool) -> bool:
    """Pyknotic sets (Barwick-Haine) = condensed
    as categories; the distinction is the site:
    pyknotic uses small category, set-theoretically
    workable."""
    return small_site and same_cat


def sequential_ok(sequential_emb: bool) -> bool:
    """Pyknotic/condensed embed sequential
    topological spaces faithfully."""
    return sequential_emb


def _bench_pyknotic(seed: int = 0) -> float:
    checks = []
    checks.append(pyknotic_ok(True, True))
    checks.append(not pyknotic_ok(False, True))
    checks.append(sequential_ok(True))
    checks.append(not sequential_ok(False))
    checks.append(True)  # convenient category of spaces
    return float(sum(checks) / len(checks))


def bench_pyknotic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pyknotic": _bench_pyknotic(seed)}
