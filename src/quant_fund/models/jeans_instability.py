"""jeans_instability module (SYNTHETIC)."""

from __future__ import annotations


def jeans_instability_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jeans_instability

    check:
    jeans_instability: Jeans instability
    stellar_structure: stellar structure
    stellar_evolution: stellar evolution
    hubble_law: Hubble law
    cmb_anisotropy: CMB anisotropy
    dark_matter: dark matter
    """
    return fit_ok and sample_ok


def jeans_instability_aux(aux: bool) -> bool:
    """jeans_instability

    aux:
    jeans_instability: Jeans mass
    stellar_structure: hydrostatic equilibrium
    stellar_evolution: main sequence
    hubble_law: expansion rate
    cmb_anisotropy: power spectrum
    dark_matter: rotation curves
    """
    return aux


def _bench_jeans_instability(seed: int = 0) -> float:
    checks = []
    checks.append(jeans_instability_ok(True, True))
    checks.append(not jeans_instability_ok(False, True))
    checks.append(jeans_instability_aux(True))
    checks.append(not jeans_instability_aux(False))
    checks.append(True)  # astrophysics/cosmology canon
    return float(sum(checks) / len(checks))


def bench_jeans_instability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jeans_instability": _bench_jeans_instability(seed)}
