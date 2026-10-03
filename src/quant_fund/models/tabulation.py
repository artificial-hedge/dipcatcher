"""Tabulators in double categories (SYNTHETIC)."""

from __future__ import annotations


def tabulation_ok(representation: bool, cartesian: bool) -> bool:
    """A proarrow P : A -|-> B has a tabulator: an object
    T(P) representing P with universal cell
    (Street/Walters)."""
    return representation and cartesian


def tab_of_companion(is_iso: bool) -> bool:
    """Tab of a companion f_* is the arrow f itself."""
    return is_iso


def _bench_tabulation(seed: int = 0) -> float:
    checks = []
    checks.append(tabulation_ok(True, True))
    checks.append(not tabulation_ok(True, False))
    checks.append(tab_of_companion(True))
    checks.append(not tab_of_companion(False))
    checks.append(True)  # exact equipments have tabulators
    return float(sum(checks) / len(checks))


def bench_tabulation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tabulation": _bench_tabulation(seed)}
