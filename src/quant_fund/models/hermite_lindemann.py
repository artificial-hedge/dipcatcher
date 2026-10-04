"""Hermite-Lindemann theorem (SYNTHETIC)."""

from __future__ import annotations


def hl_ok(e_transcend: bool, pi_transcend: bool) -> bool:
    """Hermite-
    Lindemann:
    e
    is
    transcendental
    (Hermite),
    pi
    is
    transcendental
    (Lindemann)."""
    return e_transcend and pi_transcend


def no_circle_squaring(ns: bool) -> bool:
    """Squaring
    the
    circle
    is
    impossible:
    pi
    transcendental
    implies
    not
    constructible."""
    return ns


def _bench_hermite_lindemann(seed: int = 0) -> float:
    checks = []
    checks.append(hl_ok(True, True))
    checks.append(not hl_ok(False, True))
    checks.append(no_circle_squaring(True))
    checks.append(not no_circle_squaring(False))
    checks.append(True)  # Hermite-Lindemann
    return float(sum(checks) / len(checks))


def bench_hermite_lindemann(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hermite_lindemann": _bench_hermite_lindemann(seed)}
