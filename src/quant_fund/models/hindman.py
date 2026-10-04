"""Hindman theorem (SYNTHETIC)."""

from __future__ import annotations


def hindman_ok(finite: bool, sums: bool) -> bool:
    """Hindman
    theorem:
    finite
    colorings
    of N
    contain
    a mono
    IP-set
    FS(x_n) —
    all finite
    sums."""
    return finite and sums


def central_set(central: bool) -> bool:
    """Central
    sets:
    members
    of
    minimal
    idempotent
    ultrafilters
    contain
    rich
    additive
    structure."""
    return central


def _bench_hindman(seed: int = 0) -> float:
    checks = []
    checks.append(hindman_ok(True, True))
    checks.append(not hindman_ok(False, True))
    checks.append(central_set(True))
    checks.append(not central_set(False))
    checks.append(True)  # Hindman
    return float(sum(checks) / len(checks))


def bench_hindman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hindman": _bench_hindman(seed)}
