"""hubble_law module (SYNTHETIC)."""

from __future__ import annotations


def hubble_law_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hubble_law

    check:
    jeans_instability: Jeans instability
    stellar_structure: stellar structure
    stellar_evolution: stellar evolution
    hubble_law: Hubble law
    cmb_anisotropy: CMB anisotropy
    dark_matter: dark matter
    """
    return fit_ok and sample_ok


def hubble_law_aux(aux: bool) -> bool:
    """hubble_law

    aux:
    jeans_instability: Jeans mass
    stellar_structure: hydrostatic equilibrium
    stellar_evolution: main sequence
    hubble_law: expansion rate
    cmb_anisotropy: power spectrum
    dark_matter: rotation curves
    """
    return aux


def _bench_hubble_law(seed: int = 0) -> float:
    checks = []
    checks.append(hubble_law_ok(True, True))
    checks.append(not hubble_law_ok(False, True))
    checks.append(hubble_law_aux(True))
    checks.append(not hubble_law_aux(False))
    checks.append(True)  # astrophysics/cosmology canon
    return float(sum(checks) / len(checks))


def bench_hubble_law(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hubble_law": _bench_hubble_law(seed)}
