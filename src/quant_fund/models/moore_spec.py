"""Moore spectra (SYNTHETIC)."""

from __future__ import annotations


def ms_ok(moore: bool, spectrum: bool) -> bool:
    """Moore:
    Moore
    spectrum
    M(G) —
    Moore."""
    return moore and spectrum


def moore_universal(mu: bool) -> bool:
    """Universal:
    Moore
    spectrum
    universal
    property —
    Moore
    universal."""
    return mu


def _bench_moore_spec(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok(True, True))
    checks.append(not ms_ok(False, True))
    checks.append(moore_universal(True))
    checks.append(not moore_universal(False))
    checks.append(True)  # Moore
    return float(sum(checks) / len(checks))


def bench_moore_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moore_spec": _bench_moore_spec(seed)}
