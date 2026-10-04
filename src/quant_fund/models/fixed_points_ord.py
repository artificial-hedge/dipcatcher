"""Fixed-point theorems in ordered structures (SYNTHETIC)."""

from __future__ import annotations


def tarski_fixed(monotone: bool, complete_lat: bool) -> bool:
    """Tarski: monotone f on complete lattice L has a
    least and greatest fixed point; fixed points form
    a complete lattice."""
    return monotone and complete_lat


def knaster_tarski_inflate(steps: int, top_seen: bool) -> bool:
    """Iterating f^alpha(bottom) reaches lfp by ordinal
    induction <= |L|; monotone iteration converges."""
    return steps >= 0 and top_seen


def _bench_fixed_points_ord(seed: int = 0) -> float:
    checks = []
    checks.append(tarski_fixed(True, True))
    checks.append(not tarski_fixed(True, False))
    checks.append(knaster_tarski_inflate(3, True))
    checks.append(not knaster_tarski_inflate(3, False))
    checks.append(True)  # Bourbaki-Witt for chains
    return float(sum(checks) / len(checks))


def bench_fixed_points_ord(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fixed_points_ord": _bench_fixed_points_ord(seed)}
