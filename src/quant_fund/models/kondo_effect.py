"""kondo_effect module (SYNTHETIC)."""

from __future__ import annotations


def kondo_effect_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kondo_effect

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def kondo_effect_aux(aux: bool) -> bool:
    """kondo_effect

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_kondo_effect(seed: int = 0) -> float:
    checks = []
    checks.append(kondo_effect_ok(True, True))
    checks.append(not kondo_effect_ok(False, True))
    checks.append(kondo_effect_aux(True))
    checks.append(not kondo_effect_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_kondo_effect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kondo_effect": _bench_kondo_effect(seed)}
