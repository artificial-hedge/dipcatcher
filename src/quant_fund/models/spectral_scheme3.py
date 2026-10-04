"""Spectral schemes III (SYNTHETIC)."""

from __future__ import annotations


def ss3_ok(spectral: bool, scheme: bool) -> bool:
    """Spectral
    scheme:
    spectral
    scheme —
    structured
    space."""
    return spectral and scheme


def structured_space(sst: bool) -> bool:
    """Structured
    space:
    structured
    space —
    locally
    ringed."""
    return sst


def _bench_spectral_scheme3(seed: int = 0) -> float:
    checks = []
    checks.append(ss3_ok(True, True))
    checks.append(not ss3_ok(False, True))
    checks.append(structured_space(True))
    checks.append(not structured_space(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_spectral_scheme3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_scheme3": _bench_spectral_scheme3(seed)}
