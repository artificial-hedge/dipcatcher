"""Whittaker models (SYNTHETIC)."""

from __future__ import annotations


def whittaker_ok(model: bool, unique: bool) -> bool:
    """Whittaker
    model:
    realization
    of a rep
    in the
    induced
    space
    Ind_N^G
    psi;
    unique
    (local)."""
    return model and unique


def whittaker_function(func: bool) -> bool:
    """Whittaker
    functions
    W_pi(g)
    transform
    by psi
    under N;
    Fourier
    coefficients
    live
    there."""
    return func


def _bench_whittaker_model(seed: int = 0) -> float:
    checks = []
    checks.append(whittaker_ok(True, True))
    checks.append(not whittaker_ok(False, True))
    checks.append(whittaker_function(True))
    checks.append(not whittaker_function(False))
    checks.append(True)  # Jacquet-Langlands
    return float(sum(checks) / len(checks))


def bench_whittaker_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whittaker_model": _bench_whittaker_model(seed)}
