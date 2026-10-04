"""band_structure module (SYNTHETIC)."""

from __future__ import annotations


def band_structure_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """band_structure

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def band_structure_aux(aux: bool) -> bool:
    """band_structure

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_band_structure(seed: int = 0) -> float:
    checks = []
    checks.append(band_structure_ok(True, True))
    checks.append(not band_structure_ok(False, True))
    checks.append(band_structure_aux(True))
    checks.append(not band_structure_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_band_structure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_band_structure": _bench_band_structure(seed)}
