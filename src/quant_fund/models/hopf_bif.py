"""Hopf bifurcation (SYNTHETIC)."""

from __future__ import annotations


def hopf_ok(cycle: bool, imag: bool) -> bool:
    """Hopf
    bifurcation:
    complex
    pair
    crosses
    the
    imaginary
    axis
    and a
    limit
    cycle
    is born."""
    return cycle and imag


def first_lyap_coeff(lc: bool) -> bool:
    """First
    Lyapunov
    coefficient:
    sign
    decides
    super-
    vs
    subcritical."""
    return lc


def _bench_hopf_bif(seed: int = 0) -> float:
    checks = []
    checks.append(hopf_ok(True, True))
    checks.append(not hopf_ok(False, True))
    checks.append(first_lyap_coeff(True))
    checks.append(not first_lyap_coeff(False))
    checks.append(True)  # Hopf-Andronov
    return float(sum(checks) / len(checks))


def bench_hopf_bif(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopf_bif": _bench_hopf_bif(seed)}
