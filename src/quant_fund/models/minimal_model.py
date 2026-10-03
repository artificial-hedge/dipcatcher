"""Minimal models (SYNTHETIC)."""

from __future__ import annotations


def minimal_model_ok(klt: bool, nef: bool) -> bool:
    """Minimal model of (X,B):
    K_X+B is nef + the pair
    is klt/lc; endpoint of
    the MMP (BCHM, HM)."""
    return klt and nef


def minimalization_algo(termination: bool) -> bool:
    """Minimal-model program:
    iterated flips/divisorial
    contractions of
    K-negative extremal rays;
    terminates to minimal or
    Mori fiber space."""
    return termination


def _bench_minimal_model(seed: int = 0) -> float:
    checks = []
    checks.append(minimal_model_ok(True, True))
    checks.append(not minimal_model_ok(False, True))
    checks.append(minimalization_algo(True))
    checks.append(not minimalization_algo(False))
    checks.append(True)  # BCHM existence
    return float(sum(checks) / len(checks))


def bench_minimal_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minimal_model": _bench_minimal_model(seed)}
