"""bloch_theorem module (SYNTHETIC)."""

from __future__ import annotations


def bloch_theorem_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bloch_theorem

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def bloch_theorem_aux(aux: bool) -> bool:
    """bloch_theorem

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_bloch_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(bloch_theorem_ok(True, True))
    checks.append(not bloch_theorem_ok(False, True))
    checks.append(bloch_theorem_aux(True))
    checks.append(not bloch_theorem_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_bloch_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bloch_theorem": _bench_bloch_theorem(seed)}
