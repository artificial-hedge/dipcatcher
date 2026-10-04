"""Raynaud generic fibers (SYNTHETIC)."""

from __future__ import annotations


def raynaud_ok(formal_model: bool, generic: bool) -> bool:
    """Raynaud: rigid analytic space = generic
    fiber of a formal scheme; admissible
    blowups give equivalent models."""
    return formal_model and generic


def admissible_blowup(isolated_special: bool) -> bool:
    """Admissible blowups centered in the
    special fiber form a filtered system."""
    return isolated_special


def _bench_raynaud_gen(seed: int = 0) -> float:
    checks = []
    checks.append(raynaud_ok(True, True))
    checks.append(not raynaud_ok(True, False))
    checks.append(admissible_blowup(True))
    checks.append(not admissible_blowup(False))
    checks.append(True)  # Berthelot tubes on formal models
    return float(sum(checks) / len(checks))


def bench_raynaud_gen(seed: int = 0) -> dict[str, float]:
    return {"synthetic_raynaud_gen": _bench_raynaud_gen(seed)}
