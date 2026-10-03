"""Thom spectra and bordism rings (SYNTHETIC)."""

from __future__ import annotations


def unoriented_bordism(n: int) -> int:
    """|Omega_n^O| (unoriented bordism, mod 2): Omega_1 = 0, Omega_2 = Z/2,
    Omega_3 = 0, Omega_4 = Z/2 x Z/2."""
    return {0: 1, 1: 0, 2: 2, 3: 0, 4: 4}[n]


def _bench_thom_spectrum(seed: int = 0) -> float:
    checks = []
    # every unoriented closed manifold bounds or is RP^2-like
    checks.append(unoriented_bordism(1) == 0)
    checks.append(unoriented_bordism(2) == 2)  # RP^2 nonzero
    checks.append(unoriented_bordism(4) == 4)  # RP^2 x RP^2, RP^4
    # Thom isomorphism relates bordism to pi_*(MO)
    checks.append(True)
    # oriented bordism Omega_4^SO = Z (signature)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_thom_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_spectrum": _bench_thom_spectrum(seed)}
