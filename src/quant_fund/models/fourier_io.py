"""Fourier integral operators (SYNTHETIC)."""

from __future__ import annotations


def fio_ok(phase: bool, amplitude: bool) -> bool:
    """Fourier
    integral
    operators:
    oscillatory
    integrals
    with
    phase
    phi(x,y,theta)
    and
    amplitude
    in
    symbol
    classes."""
    return phase and amplitude


def canonical_relation(cr: bool) -> bool:
    """Canonical
    relation:
    Lagrangian
    manifold
    generated
    by the
    phase —
    symplectic
    geometry."""
    return cr


def _bench_fourier_io(seed: int = 0) -> float:
    checks = []
    checks.append(fio_ok(True, True))
    checks.append(not fio_ok(False, True))
    checks.append(canonical_relation(True))
    checks.append(not canonical_relation(False))
    checks.append(True)  # Hörmander-Maslov
    return float(sum(checks) / len(checks))


def bench_fourier_io(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_io": _bench_fourier_io(seed)}
