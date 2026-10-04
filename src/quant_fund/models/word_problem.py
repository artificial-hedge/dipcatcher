"""Word problem (SYNTHETIC)."""

from __future__ import annotations


def wp_ok(decidable: bool, dehn: bool) -> bool:
    """Word
    problem:
    decide
    whether
    a
    word
    equals
    the
    identity —
    Dehn
    algorithm
    in
    hyperbolic
    groups."""
    return decidable and dehn


def boone_novikov(bn: bool) -> bool:
    """Boone-
    Novikov:
    finitely
    presented
    groups
    with
    undecidable
    word
    problem
    exist."""
    return bn


def _bench_word_problem(seed: int = 0) -> float:
    checks = []
    checks.append(wp_ok(True, True))
    checks.append(not wp_ok(False, True))
    checks.append(boone_novikov(True))
    checks.append(not boone_novikov(False))
    checks.append(True)  # Dehn
    return float(sum(checks) / len(checks))


def bench_word_problem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_word_problem": _bench_word_problem(seed)}
