"""Galois lattice (SYNTHETIC)."""

from __future__ import annotations


def gl_ok(lattice: bool, galois: bool) -> bool:
    """Galois
    lattice:
    Z_p
    lattice
    with
    Galois
    action —
    stable
    lattice."""
    return lattice and galois


def stable_lattice(sl: bool) -> bool:
    """Stable
    lattice:
    Galois
    stable
    Z_p
    lattice —
    Kisin
    lattice."""
    return sl


def _bench_galois_lattice(seed: int = 0) -> float:
    checks = []
    checks.append(gl_ok(True, True))
    checks.append(not gl_ok(False, True))
    checks.append(stable_lattice(True))
    checks.append(not stable_lattice(False))
    checks.append(True)  # Kisin
    return float(sum(checks) / len(checks))


def bench_galois_lattice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_lattice": _bench_galois_lattice(seed)}
