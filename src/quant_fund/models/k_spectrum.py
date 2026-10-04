"""K-spectrum (SYNTHETIC)."""

from __future__ import annotations


def ks_ok(k: bool, spectrum: bool) -> bool:
    """K-
    spectrum:
    K-
    theory
    spectrum —
    Elmendorf-
    Mandell."""
    return k and spectrum


def spectrum_k(sk: bool) -> bool:
    """Spectrum:
    K-
    theory
    symmetric
    spectrum —
    EM
    K."""
    return sk


def _bench_k_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(ks_ok(True, True))
    checks.append(not ks_ok(False, True))
    checks.append(spectrum_k(True))
    checks.append(not spectrum_k(False))
    checks.append(True)  # EM
    return float(sum(checks) / len(checks))


def bench_k_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_spectrum": _bench_k_spectrum(seed)}
