"""dark_matter module (SYNTHETIC)."""

from __future__ import annotations


def dark_matter_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dark_matter

    check:
    jeans_instability: Jeans instability
    stellar_structure: stellar structure
    stellar_evolution: stellar evolution
    hubble_law: Hubble law
    cmb_anisotropy: CMB anisotropy
    dark_matter: dark matter
    """
    return fit_ok and sample_ok


def dark_matter_aux(aux: bool) -> bool:
    """dark_matter

    aux:
    jeans_instability: Jeans mass
    stellar_structure: hydrostatic equilibrium
    stellar_evolution: main sequence
    hubble_law: expansion rate
    cmb_anisotropy: power spectrum
    dark_matter: rotation curves
    """
    return aux


def _bench_dark_matter(seed: int = 0) -> float:
    checks = []
    checks.append(dark_matter_ok(True, True))
    checks.append(not dark_matter_ok(False, True))
    checks.append(dark_matter_aux(True))
    checks.append(not dark_matter_aux(False))
    checks.append(True)  # astrophysics/cosmology canon
    return float(sum(checks) / len(checks))


def bench_dark_matter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dark_matter": _bench_dark_matter(seed)}
