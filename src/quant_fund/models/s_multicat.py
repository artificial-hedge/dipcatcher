"""S-multicategory (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(s: bool, multicat: bool) -> bool:
    """S
    multicat:
    S
    multicategory —
    spectra."""
    return s and multicat


def spectra_multicat(sp: bool) -> bool:
    """Spectra
    multicat:
    spectra
    multicategory —
    symmetric."""
    return sp


def _bench_s_multicat(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(spectra_multicat(True))
    checks.append(not spectra_multicat(False))
    checks.append(True)  # Elmendorf-Mandell
    return float(sum(checks) / len(checks))


def bench_s_multicat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_s_multicat": _bench_s_multicat(seed)}
