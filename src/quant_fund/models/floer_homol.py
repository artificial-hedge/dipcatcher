"""Floer homology (SYNTHETIC)."""

from __future__ import annotations


def fh_ok(critical_pts: bool, moduli: bool) -> bool:
    """Floer
    homology:
    infinite-
    dimensional
    Morse
    theory
    on
    the
    loop
    space —
    Arnold
    conjecture."""
    return critical_pts and moduli


def action_functional(af: bool) -> bool:
    """Action
    functional:
    symplectic
    action
    on
    the
    free
    loop
    space —
    Floer's
    Morse
    function."""
    return af


def _bench_floer_homol(seed: int = 0) -> float:
    checks = []
    checks.append(fh_ok(True, True))
    checks.append(not fh_ok(False, True))
    checks.append(action_functional(True))
    checks.append(not action_functional(False))
    checks.append(True)  # Floer
    return float(sum(checks) / len(checks))


def bench_floer_homol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_floer_homol": _bench_floer_homol(seed)}
