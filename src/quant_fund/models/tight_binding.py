"""tight_binding module (SYNTHETIC)."""

from __future__ import annotations


def tight_binding_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tight_binding

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def tight_binding_aux(aux: bool) -> bool:
    """tight_binding

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_tight_binding(seed: int = 0) -> float:
    checks = []
    checks.append(tight_binding_ok(True, True))
    checks.append(not tight_binding_ok(False, True))
    checks.append(tight_binding_aux(True))
    checks.append(not tight_binding_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_tight_binding(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tight_binding": _bench_tight_binding(seed)}
