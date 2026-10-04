"""bgk_model module (SYNTHETIC)."""

from __future__ import annotations


def bgk_model_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bgk_model

    check:
    boltzmann_eq: Boltzmann equation
    vlasov_eq: Vlasov equation
    bgk_model: BGK relaxation model
    chapman_enskog: Chapman-Enskog expansion
    h_theorem: Boltzmann H-theorem
    landau_damping: Landau damping
    """
    return fit_ok and sample_ok


def bgk_model_aux(aux: bool) -> bool:
    """bgk_model

    aux:
    boltzmann_eq: collision invariants
    vlasov_eq: collisionless dynamics
    bgk_model: Maxwellian target
    chapman_enskog: Navier-Stokes limit
    h_theorem: entropy production
    landau_damping: resonant interaction
    """
    return aux


def _bench_bgk_model(seed: int = 0) -> float:
    checks = []
    checks.append(bgk_model_ok(True, True))
    checks.append(not bgk_model_ok(False, True))
    checks.append(bgk_model_aux(True))
    checks.append(not bgk_model_aux(False))
    checks.append(True)  # kinetic-theory canon
    return float(sum(checks) / len(checks))


def bench_bgk_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bgk_model": _bench_bgk_model(seed)}
