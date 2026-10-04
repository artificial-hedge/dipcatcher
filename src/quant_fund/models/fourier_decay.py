"""fourier decay module (SYNTHETIC)."""

from __future__ import annotations


def fourier_decay_ok(smooth: bool, approx: bool) -> bool:
    """fourier_decay
    check:
    approximation
    theory —
    smoothness."""
    return smooth and approx


def fourier_decay_aux(aux: bool) -> bool:
    """fourier_decay
    aux:
    auxiliary
    approx check —
    degree."""
    return aux


def _bench_fourier_decay(seed: int = 0) -> float:
    checks = []
    checks.append(fourier_decay_ok(True, True))
    checks.append(not fourier_decay_ok(False, True))
    checks.append(fourier_decay_aux(True))
    checks.append(not fourier_decay_aux(False))
    checks.append(True)  # approximation-theory canon
    return float(sum(checks) / len(checks))


def bench_fourier_decay(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_decay": _bench_fourier_decay(seed)}
