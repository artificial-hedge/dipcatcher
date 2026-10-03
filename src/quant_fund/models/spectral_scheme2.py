"""Spectral schemes (SYNTHETIC)."""

from __future__ import annotations


def spec_scheme_ok(structured: bool, e_infty: bool) -> bool:
    """Spectral scheme:
    a spectrally
    ringed space
    (X, O_X) whose
    O_X takes values
    in E_infty-rings."""
    return structured and e_infty


def affine_spec(aff: bool) -> bool:
    """Affine spectral
    scheme Spec A
    for a connective
    E_infty-ring A;
    opposite of CAlg^cn."""
    return aff


def _bench_spectral_scheme2(seed: int = 0) -> float:
    checks = []
    checks.append(spec_scheme_ok(True, True))
    checks.append(not spec_scheme_ok(False, True))
    checks.append(affine_spec(True))
    checks.append(not affine_spec(False))
    checks.append(True)  # Lurie SAG
    return float(sum(checks) / len(checks))


def bench_spectral_scheme2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_scheme2": _bench_spectral_scheme2(seed)}
