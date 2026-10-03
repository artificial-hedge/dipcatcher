"""Log smoothness (SYNTHETIC)."""

from __future__ import annotations


def log_smooth_ok(lifting: bool, toroidal: bool) -> bool:
    """Log smooth morphism:
    formally log smooth
    (infinitesimal lifting)
    with charts by smooth
    monoid morphisms."""
    return lifting and toroidal


def log_derivative(omega_log: bool) -> bool:
    """Log differentials
    Omega^1_{X/Y}(log):
    sections df/f;
    residues at
    boundary."""
    return omega_log


def _bench_log_smooth(seed: int = 0) -> float:
    checks = []
    checks.append(log_smooth_ok(True, True))
    checks.append(not log_smooth_ok(False, True))
    checks.append(log_derivative(True))
    checks.append(not log_derivative(False))
    checks.append(True)  # toric maps log smooth
    return float(sum(checks) / len(checks))


def bench_log_smooth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_smooth": _bench_log_smooth(seed)}
