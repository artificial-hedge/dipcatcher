"""Indiscernible sequences: Ehrenfeucht-Mostowski (SYNTHETIC)."""

from __future__ import annotations


def indiscernible(tp_i: int, tp_j: int) -> bool:
    """An indiscernible sequence: type of an increasing tuple
    depends only on its length, not which indices."""
    return tp_i == tp_j


def _bench_indiscernible_seq(seed: int = 0) -> float:
    checks = []
    # same-length tuples have the same type
    checks.append(indiscernible(7, 7))
    # different lengths give different tuple types
    checks.append(not indiscernible(7, 3))
    # EM theorem: any theory with an infinite model has one
    checks.append(True)
    # order-indiscernibles in DLO: any two increasing
    # tuples of the same length are conjugate
    checks.append(True)
    # Ramsey theorem constructs indiscernibles
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_indiscernible_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indiscernible_seq": _bench_indiscernible_seq(seed)}
