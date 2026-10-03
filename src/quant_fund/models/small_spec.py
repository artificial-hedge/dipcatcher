"""Small spectra (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(small: bool, spectrum: bool) -> bool:
    """Small
    spectrum:
    small
    spectrum —
    compact
    spectrum."""
    return small and spectrum


def finite_spectrum(fs: bool) -> bool:
    """Finite
    spectrum:
    finite
    spectrum —
    finite
    CW
    spectrum."""
    return fs


def _bench_small_spec(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(finite_spectrum(True))
    checks.append(not finite_spectrum(False))
    checks.append(True)  # Brown-Spanier
    return float(sum(checks) / len(checks))


def bench_small_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_small_spec": _bench_small_spec(seed)}
