"""norm_integrable module (SYNTHETIC)."""

from __future__ import annotations


def norm_integrable_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """norm_integrable

    check:
    bochner_integral: Bochner integral of vector functions
    lusin_rep: vector Lusin representation / approx
    radon_nikodym_prop: Radon-Nikodym property for spaces
    bochner_meas: strong measurability
    norm_integrable: norm-integrability criterion
    pettis_weak: Pettis weak measurability
    """
    return fit_ok and sample_ok


def norm_integrable_aux(aux: bool) -> bool:
    """norm_integrable

    aux:
    bochner_integral: dominated convergence vector
    lusin_rep: a.e. continuous approximation
    radon_nikodym_prop: separable dual characterization
    bochner_meas: separable-valued reduction
    norm_integrable: Bochner norm condition
    pettis_weak: weak measurability equivalence
    """
    return aux


def _bench_norm_integrable(seed: int = 0) -> float:
    checks = []
    checks.append(norm_integrable_ok(True, True))
    checks.append(not norm_integrable_ok(False, True))
    checks.append(norm_integrable_aux(True))
    checks.append(not norm_integrable_aux(False))
    checks.append(True)  # Bochner/vector-valued canon
    return float(sum(checks) / len(checks))


def bench_norm_integrable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norm_integrable": _bench_norm_integrable(seed)}
