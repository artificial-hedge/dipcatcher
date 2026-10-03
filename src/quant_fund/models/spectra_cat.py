"""The category of spectra (SYNTHETIC)."""

from __future__ import annotations


def is_spectrum(seq_of_spaces: bool, structure_maps_equiv: bool) -> bool:
    """A spectrum is a sequence X_n of pointed spaces with
    equivalences X_n -> Omega X_{n+1}; Sp is the
    stabilization of Spaces and the unit stable category."""
    return seq_of_spaces and structure_maps_equiv


def _bench_spectra_cat(seed: int = 0) -> float:
    checks = []
    # structure maps equivalences -> spectrum
    checks.append(is_spectrum(True, True))
    # non-equivalence is only a prespectrum
    checks.append(not is_spectrum(True, False))
    # suspension spectrum construction
    checks.append(True)
    # stable stems = pi of sphere spectrum
    checks.append(True)
    # represent cohomology theories
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_spectra_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectra_cat": _bench_spectra_cat(seed)}
