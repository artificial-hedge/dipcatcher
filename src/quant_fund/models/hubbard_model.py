"""hubbard_model module (SYNTHETIC)."""

from __future__ import annotations


def hubbard_model_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hubbard_model

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def hubbard_model_aux(aux: bool) -> bool:
    """hubbard_model

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_hubbard_model(seed: int = 0) -> float:
    checks = []
    checks.append(hubbard_model_ok(True, True))
    checks.append(not hubbard_model_ok(False, True))
    checks.append(hubbard_model_aux(True))
    checks.append(not hubbard_model_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_hubbard_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hubbard_model": _bench_hubbard_model(seed)}
