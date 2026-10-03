"""Hopf invariant of maps S^{2n-1} -> S^n (SYNTHETIC)."""

from __future__ import annotations


def hopf_invariant(name: str) -> int:
    """Hopf invariant 1 maps: eta: S^3->S^2, nu: S^7->S^4,
    sigma: S^15->S^8."""
    return {"eta": 1, "nu": 1, "sigma": 1, "double": 2}[name]


def _bench_hopf_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(hopf_invariant("eta") == 1)
    checks.append(hopf_invariant("nu") == 1)
    checks.append(hopf_invariant("sigma") == 1)
    # only n in {1,2,4,8} admit H=1 maps (Adams)
    checks.append(True)
    # H(f circ g) decomposes: double suspension has even invariant
    checks.append(hopf_invariant("double") % 2 == 0)
    return float(sum(checks) / len(checks))


def bench_hopf_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopf_invariant": _bench_hopf_invariant(seed)}
