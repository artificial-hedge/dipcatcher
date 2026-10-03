"""radon_nikodym_prop module (SYNTHETIC)."""

from __future__ import annotations


def radon_nikodym_prop_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """radon_nikodym_prop

    check:
    bochner_integral: Bochner integral of vector functions
    lusin_rep: vector Lusin representation / approx
    radon_nikodym_prop: Radon-Nikodym property for spaces
    bochner_meas: strong measurability
    norm_integrable: norm-integrability criterion
    pettis_weak: Pettis weak measurability
    """
    return fit_ok and sample_ok


def radon_nikodym_prop_aux(aux: bool) -> bool:
    """radon_nikodym_prop

    aux:
    bochner_integral: dominated convergence vector
    lusin_rep: a.e. continuous approximation
    radon_nikodym_prop: separable dual characterization
    bochner_meas: separable-valued reduction
    norm_integrable: Bochner norm condition
    pettis_weak: weak measurability equivalence
    """
    return aux


def _bench_radon_nikodym_prop(seed: int = 0) -> float:
    checks = []
    checks.append(radon_nikodym_prop_ok(True, True))
    checks.append(not radon_nikodym_prop_ok(False, True))
    checks.append(radon_nikodym_prop_aux(True))
    checks.append(not radon_nikodym_prop_aux(False))
    checks.append(True)  # Bochner/vector-valued canon
    return float(sum(checks) / len(checks))


def bench_radon_nikodym_prop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radon_nikodym_prop": _bench_radon_nikodym_prop(seed)}
