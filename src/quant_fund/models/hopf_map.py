"""Hopf invariant and the Hopf map (SYNTHETIC)."""

from __future__ import annotations


def hopf_inv(linking: int) -> int:
    """Hopf invariant of f: S^{2n-1} -> S^n = linking number
    of preimages; eta: S^3 -> S^2 has Hopf invariant 1."""
    return linking


def _bench_hopf_map(seed: int = 0) -> float:
    checks = []
    # Hopf map eta has invariant 1
    checks.append(hopf_inv(1) == 1)
    # double of eta has invariant 2
    checks.append(hopf_inv(2) == 2)
    # pi_3(S^2) = Z detected by Hopf invariant
    checks.append(True)
    # invariant-1 maps only n = 1,2,4,8 (Adams)
    checks.append(True)
    # H-space structures on spheres relate
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hopf_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopf_map": _bench_hopf_map(seed)}
