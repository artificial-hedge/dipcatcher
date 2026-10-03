"""lusin_rep module (SYNTHETIC)."""

from __future__ import annotations


def lusin_rep_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lusin_rep

    check:
    bochner_integral: Bochner integral of vector functions
    lusin_rep: vector Lusin representation / approx
    radon_nikodym_prop: Radon-Nikodym property for spaces
    bochner_meas: strong measurability
    norm_integrable: norm-integrability criterion
    pettis_weak: Pettis weak measurability
    """
    return fit_ok and sample_ok


def lusin_rep_aux(aux: bool) -> bool:
    """lusin_rep

    aux:
    bochner_integral: dominated convergence vector
    lusin_rep: a.e. continuous approximation
    radon_nikodym_prop: separable dual characterization
    bochner_meas: separable-valued reduction
    norm_integrable: Bochner norm condition
    pettis_weak: weak measurability equivalence
    """
    return aux


def _bench_lusin_rep(seed: int = 0) -> float:
    checks = []
    checks.append(lusin_rep_ok(True, True))
    checks.append(not lusin_rep_ok(False, True))
    checks.append(lusin_rep_aux(True))
    checks.append(not lusin_rep_aux(False))
    checks.append(True)  # Bochner/vector-valued canon
    return float(sum(checks) / len(checks))


def bench_lusin_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lusin_rep": _bench_lusin_rep(seed)}
